"""
This module contains the MqttClient class, which handles external MQTT messaging
"""

import json
import logging
import uuid
from datetime import datetime
from queue import Queue
from threading import Lock
from typing import Any

import paho.mqtt.client as mqtt

from freqtrade.constants import PairWithTimeframe
from freqtrade.enums.rpcmessagetype import RPCMessageType
from freqtrade.exceptions import OperationalException
from freqtrade.rpc import RPCHandler
from freqtrade.rpc.rpc import RPC
from freqtrade.rpc.rpc_types import RPCSendMsg


logger = logging.getLogger(__name__)


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects"""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class MqttConnection:
    """
    Represents a single MQTT connection in the pool
    """

    def __init__(self, config: dict[str, Any], short_uuid: str, connection_num: int):
        mqtt_config = config.get("external_mqtt_server", {})
        client_id_prefix = mqtt_config.get("client_id_prefix", "freqtrade_")
        client_id = f"{client_id_prefix}{short_uuid}_{connection_num}"

        self.client = mqtt.Client(client_id=client_id, reconnect_on_failure=True)
        self.config = config
        self.in_use = False
        self.setup_client()

    def setup_client(self):
        """Configure and connect the MQTT client"""
        mqtt_config = self.config.get("external_mqtt_server", {})
        self.client.username_pw_set(mqtt_config.get("username"), mqtt_config.get("password"))

        if mqtt_config.get("protocol", "").lower() == "tls":
            self.client.tls_set()

        try:
            self.client.connect(mqtt_config.get("host", "localhost"), mqtt_config.get("port", 1883))
            self.client.loop_start()
        except Exception as e:
            logger.error(f"Error connecting to MQTT broker: {e}")
            raise


class MqttClient(RPCHandler):
    """
    MQTT client class handling message publication to an external MQTT broker
    """

    # Initialize class variables
    _has_rpc: bool = False
    _rpc: RPC = None

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize MQTT client
        """
        self._config = config
        self._name = "external_mqtt_server"

        self._pool_size = config.get("external_mqtt_server", {}).get("pool_size", 10)
        self._connection_pool: list[MqttConnection] = []
        self._pool_lock = Lock()
        self._message_queue: Queue = Queue()

        self._mqtt_config = self._config.get("external_mqtt_server", {})
        self._exchange_name = self._config.get("exchange", {}).get("name", "")
        self._retain: bool = self._mqtt_config.get("retain", False)
        self._short_uuid = str(uuid.uuid4())[:8]

        # Add tracking for sent records - using list instead of set to maintain order
        self._sent_records: dict[str, list] = {}
        self._max_records_per_topic = 3  # Keep only last 3 records

        # Initialize connection pool
        self._initialize_pool()

    def _initialize_pool(self) -> None:
        """Initialize the connection pool with MQTT clients"""
        try:
            for i in range(self._pool_size):
                connection = MqttConnection(self._config, self._short_uuid, i)
                self._connection_pool.append(connection)
        except Exception as e:
            logger.error(f"Failed to initialize MQTT connection pool: {e}")
            raise

    def _get_connection(self) -> MqttConnection | None:
        """Get an available connection from the pool"""
        with self._pool_lock:
            for conn in self._connection_pool:
                if not conn.in_use:
                    conn.in_use = True
                    return conn
        return None

    def _release_connection(self, connection: MqttConnection) -> None:
        """Release a connection back to the pool"""
        with self._pool_lock:
            connection.in_use = False

    def add_rpc_handler(self, rpc: RPC):
        """
        Attach rpc handler
        """
        if not MqttClient._has_rpc:
            MqttClient._rpc = rpc
            MqttClient._has_rpc = True
        else:
            # This should not happen assuming we didn't mess up.
            raise OperationalException("RPC Handler already attached.")

    def cleanup(self) -> None:
        """
        Cleanup pending connections and messages
        """
        logger.info("Cleaning up MQTT connections...")
        for conn in self._connection_pool:
            try:
                conn.client.loop_stop()
                conn.client.disconnect()
            except Exception as e:
                logger.error(f"Error during MQTT cleanup: {e}")

    def _is_record_sent(self, topic: str, record: dict) -> bool:
        """
        Check if a record has already been sent
        Returns True if record was already sent, False otherwise
        Maintains only the last N records per topic
        """
        # Create a unique key for the record based on timestamp and other relevant fields
        record_key = f"{record.get('date')}_{record.get('open')}_{record.get('close')}"

        if topic not in self._sent_records:
            self._sent_records[topic] = []

        # Check if record exists in the last N records
        if record_key in self._sent_records[topic]:
            return True

        # Add new record and maintain only last N records
        self._sent_records[topic].append(record_key)
        if len(self._sent_records[topic]) > self._max_records_per_topic:
            self._sent_records[topic].pop(0)  # Remove oldest record

        return False

    def send_msg(self, msg: RPCSendMsg) -> None:
        """
        Send given message to MQTT broker
        """
        if msg.get("type") == RPCMessageType.ANALYZED_DF:
            pairWmetadata: PairWithTimeframe = msg["data"]["key"]
            renamed_pair = pairWmetadata[0].replace("/", "-")
            records: list = msg["data"]["df"].to_dict(orient="records")

            base_topic: str = f"crypto/{self._exchange_name}/{pairWmetadata[1]}/{renamed_pair}"

            connection = self._get_connection()
            if not connection:
                print("No available MQTT connections in pool. Message queued.")
                logger.warning("No available MQTT connections in pool. Message queued.")
                self._message_queue.put((base_topic, msg, self._retain))
                return

            try:
                for record in records:
                    # Skip if record was already sent
                    if self._is_record_sent(base_topic, record):
                        continue

                    indicator_msg = {
                        "pair": renamed_pair,
                        "exchange": self._exchange_name,
                        "timeframe": pairWmetadata[1],
                        **record,
                    }

                    # Publish the message with custom JSON encoder
                    connection.client.publish(
                        base_topic, json.dumps(indicator_msg, cls=DateTimeEncoder), qos=1,
                        retain=self._retain
                    )
                    logger.debug(f"Published indicators to MQTT topic {base_topic}")

            except Exception as e:
                print(f"Error publishing message to MQTT: {e}")
                logger.error(f"Error publishing message to MQTT: {e}")
            finally:
                self._release_connection(connection)

            # Process queued messages if any
            while not self._message_queue.empty():
                try:
                    topic, queued_msg, retain = self._message_queue.get_nowait()
                    self.send_msg(queued_msg)
                except Exception as e:
                    print(f"Error processing queued MQTT message: {e}")
                    logger.error(f"Error processing queued MQTT message: {e}")

    @property
    def name(self) -> str:
        """Returns the name of the handler"""
        return self._name


# """
# {
#     "external_mqtt_server": {
#         "enabled": true,
#         "host": "your.mqtt.broker.com",
#         "port": 8883,
#         "protocol": "tls",
#         "username": "your_username",
#         "password": "your_password",
#         "topic": "freqtrade/messages",
#         "retain": false,
#         "pool_size": 10,
#         "client_id_prefix": "freqtrade_",
#         "allow_custom_messages": true
#     }
# }
# """
