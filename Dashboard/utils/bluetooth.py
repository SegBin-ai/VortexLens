import asyncio
import usb.core
from bleak import BleakScanner

async def get_bluetooth_devices():
    devices = await BleakScanner.discover()
    return [{'label': f'Bluetooth Device {device.name} ({device.address})', 'value': f'bt_{device.address}'} for device in devices]

def get_usb_devices():
    try:
        devices = []
        usb_devices = usb.core.find(find_all=True)
        for device in usb_devices:
            devices.append({'label': f'USB Device {device.idVendor}:{device.idProduct}', 'value': f'usb_{device.idVendor}_{device.idProduct}'})
        return devices
    except usb.core.NoBackendError:
        return [{'label': 'No USB Backend Available', 'value': 'no_backend'}]

def get_usb_bluetooth_devices():
    usb_devices = get_usb_devices()
    bluetooth_devices = asyncio.run(get_bluetooth_devices())
    return usb_devices + bluetooth_devices
