# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Helper to scan all motors connected to a MotorsBus port.

This script will scan the specified port at all supported baudrates and list
all motors that respond, along with their IDs and model numbers.

Example:

```shell
lerobot-scan-motors --port=COM7
```

Or for Linux/macOS:

```shell
lerobot-scan-motors --port=/dev/tty.usbmodem58760431551
```
"""

import logging
from dataclasses import dataclass
from pprint import pformat

import draccus

from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


@dataclass
class ScanMotorsConfig:
    port: str
    """Serial/USB port to scan (e.g., COM7 on Windows or /dev/tty.usbmodem... on Linux/macOS)"""


@draccus.wrap()
def scan_motors(cfg: ScanMotorsConfig):
    """Scan the specified port for all connected motors."""
    init_logging()
    
    print(f"\nScanning port '{cfg.port}' for motors...")
    print("This may take a few moments as we test all supported baudrates.\n")
    
    try:
        baudrate_ids = FeetechMotorsBus.scan_port(cfg.port)
        
        if not baudrate_ids:
            print("\n❌ No motors found on this port.")
            print("\nPossible reasons:")
            print("  - No motors are connected")
            print("  - Motors are not powered on")
            print("  - Wrong port specified")
            print("  - Motors are using a baudrate not in the supported list")
            return
        
        print("\n" + "="*60)
        print("✅ MOTORS FOUND:")
        print("="*60)
        
        for baudrate, motor_ids in baudrate_ids.items():
            print(f"\n📡 Baudrate: {baudrate}")
            print(f"   Found {len(motor_ids)} motor(s):")
            
            # Connect to get model numbers
            bus = FeetechMotorsBus(port=cfg.port, motors={})
            bus._connect(handshake=False)
            bus.set_baudrate(baudrate)
            
            # Get model numbers for each ID
            id_models = bus.broadcast_ping()
            bus.port_handler.closePort()
            
            if id_models:
                for motor_id, model_number in sorted(id_models.items()):
                    # Try to get model name from model number
                    from lerobot.motors.feetech.tables import MODEL_NUMBER_TABLE
                    model_name = None
                    for name, num in MODEL_NUMBER_TABLE.items():
                        if num == model_number:
                            model_name = name
                            break
                    
                    if model_name:
                        print(f"   - Motor ID {motor_id}: {model_name} (model #{model_number})")
                    else:
                        print(f"   - Motor ID {motor_id}: Unknown model (model #{model_number})")
            else:
                for motor_id in sorted(motor_ids):
                    print(f"   - Motor ID {motor_id}: (model number unavailable)")
        
        print("\n" + "="*60)
        print("\n💡 Tips:")
        print("  - If you see motors at multiple baudrates, use the one with the most motors")
        print("  - Expected motor IDs for SO Follower: 1-6")
        print("  - If some motors are missing, check their connections and power")
        print("  - Run 'lerobot-setup-motors' to configure motor IDs if needed")
        
    except Exception as e:
        print(f"\n❌ Error scanning port: {e}")
        print("\nPossible issues:")
        print("  - Port is already in use by another program")
        print("  - Port does not exist")
        print("  - Permission denied (on Linux, try: sudo chmod 666 /dev/ttyXXX)")
        raise


def main():
    scan_motors()


if __name__ == "__main__":
    main()

