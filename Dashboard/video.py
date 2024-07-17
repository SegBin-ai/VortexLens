from dash import Dash, dcc, html, Input, Output, State, exceptions
import dash_bootstrap_components as dbc
from utils.styles import styles
from layouts.upload import upload_layout, register_upload_callbacks
from layouts.gallery import gallery_layout, register_gallery_callbacks
from layouts.devices import devices_layout, register_devices_callbacks
from utils.google_cloud import get_all_structures

# Initialize the Dash app with suppress_callback_exceptions
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Fetch initial list of structures
initial_structures = get_all_structures()

# Layout of the Dash app
app.layout = dbc.Container([
    dcc.Store(id='video-url-store'),
    dcc.Store(id='device-status-store', data={'status': 'disconnected', 'name': ''}),
    dcc.Store(id='structure-list-store', data=initial_structures),
    dcc.Store(id='selected-structure-store'),
    dcc.Store(id='folder-path-store', data=''),  # Store the current folder path
    dbc.Tabs([
        dbc.Tab(label='Upload', tab_id='upload-tab'),
        dbc.Tab(label='Gallery', tab_id='gallery-tab'),
        dbc.Tab(label='Devices', tab_id='devices-tab')
    ], id='tabs', active_tab='upload-tab'),
    
    html.Div(id='tab-content', style=styles['container']),
    html.Div(id='device-status', style={'position': 'fixed', 'bottom': '10px', 'right': '10px'})
])

@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'active_tab')
)
def render_tab_content(active_tab):
    if active_tab == 'upload-tab':
        return upload_layout
    elif active_tab == 'gallery-tab':
        return gallery_layout
    elif active_tab == 'devices-tab':
        return devices_layout

# Register callbacks from other modules
register_upload_callbacks(app)
register_gallery_callbacks(app)
register_devices_callbacks(app)

if __name__ == '__main__':
    app.run_server(debug=True)
