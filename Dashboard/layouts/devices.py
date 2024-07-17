from dash import dcc, html, Input, Output, State, exceptions
import dash_bootstrap_components as dbc
from utils.bluetooth import get_usb_bluetooth_devices
from utils.styles import styles

# Layout for the Devices page
devices_layout = html.Div([
    dbc.Row([
        dbc.Col(html.H1("Devices", style=styles['header']))
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='device-dropdown',
                options=get_usb_bluetooth_devices(),
                placeholder="Select a device",
                style=styles['dropdown']
            ),
            dbc.Button('Connect', id='connect-button', style=styles['button']),
            html.Div(id='connected-text', style=styles['connected-text'])
        ], width=6)
    ])
])

def register_devices_callbacks(app):
    @app.callback(
        [Output('connected-text', 'children'),
         Output('device-status-store', 'data')],
        Input('connect-button', 'n_clicks'),
        State('device-dropdown', 'value')
    )
    def connect_device(n_clicks, selected_device):
        if n_clicks is None:
            raise exceptions.PreventUpdate
        if selected_device:
            device_status = {'status': 'connected', 'name': selected_device, 'health': 'good'}  # Example status
            return f"Connected to {selected_device}", device_status
        return '', {'status': 'disconnected', 'name': ''}

    @app.callback(
        Output('device-status', 'children'),
        Output('device-status', 'style'),
        Input('device-status-store', 'data')
    )
    def update_device_status(device_status):
        if device_status['status'] == 'connected':
            return f"Device: {device_status['name']} (Health: {device_status['health']})", {'color': 'green'}
        return "Device: Disconnected", {'color': 'red'}
