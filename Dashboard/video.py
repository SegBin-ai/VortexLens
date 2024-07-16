from utils.bluetooth import get_usb_bluetooth_devices, get_bluetooth_devices, get_usb_devices
from dash import Dash, dcc, html, Input, Output, State, exceptions
import dash_bootstrap_components as dbc
import base64
from utils.styles import styles
from utils.google_cloud import upload_to_gcs, fetch_videos_metadata

# Initialize the Dash app with suppress_callback_exceptions
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Layout of the Dash app
app.layout = dbc.Container([
    dcc.Store(id='video-url-store'),
    dbc.Tabs([
        dbc.Tab(label='Upload', tab_id='upload-tab'),
        dbc.Tab(label='Gallery', tab_id='gallery-tab'),
        dbc.Tab(label='Devices', tab_id='devices-tab')
    ], id='tabs', active_tab='upload-tab'),
    
    html.Div(id='tab-content', style=styles['container'])
])

upload_layout = html.Div([
    dbc.Row([
        dbc.Col(html.H1("TerraVortex", style=styles['header'])),
        dbc.Col(html.Img(src='https://raw.githubusercontent.com/SegBin-ai/VortexLens/windows-edition/Dashboard/Logo.png', style={'width': '150px', 'height': '150px'}), width="auto")
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Label("Upload Video"),
            dcc.Upload(
                id='upload-video',
                children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                style=styles['upload'],
                multiple=False,
                accept=".mp4,.mov"
            ),
            html.Div(id='output-filename', style={'margin-top': '10px'}),
            dbc.Label("Title"),
            dbc.Input(id='video-title', type='text', placeholder='Enter video title', style=styles['input']),
            dbc.Label("Description"),
            dbc.Textarea(id='video-description', placeholder='Enter video description', style=styles['textarea']),
            dbc.Button('Submit', id='submit-button', style=styles['button'])
        ], width=6)
    ]),
    dbc.Row([
        dbc.Col([
            html.H3(id='output-title', className="mt-3"),
            html.P(id='output-description'),
            html.Div(id='video-player-container', style=styles['video'])
        ])
    ])
])

gallery_layout = html.Div([
    html.H1("Video Gallery", style=styles['header']),
    html.Div(id='gallery-content', style=styles['gallery'])
])

devices_layout = html.Div([
    dbc.Row([
        dbc.Col(html.H1("Devices", style=styles['header']))
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='device-dropdown',
                options=get_usb_bluetooth_devices(),
                placeholder="Select a device"
            ),
            dbc.Button('Connect', id='connect-button', style=styles['button']),
            html.Div(id='connected-text', style=styles['connected-text'])
        ], width=6)
    ])
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

# Callback to update filename immediately after file is uploaded
@app.callback(
    Output('output-filename', 'children'),
    Input('upload-video', 'filename')
)
def update_filename(filename):
    if filename is not None:
        return f"Uploaded file: {filename}"
    return 'No file uploaded.'

# Callback to handle video upload and display title, description, and video
@app.callback(
    [Output('output-title', 'children'),
     Output('output-description', 'children'),
     Output('video-url-store', 'data')],
    [Input('submit-button', 'n_clicks')],
    [State('video-title', 'value'),
     State('video-description', 'value'),
     State('upload-video', 'contents'),
     State('upload-video', 'filename')]
)
def update_output(n_clicks, title, description, video_content, filename):
    if n_clicks is None:
        raise exceptions.PreventUpdate

    if video_content is not None:
        content_type, content_string = video_content.split(',')
        video_data = base64.b64decode(content_string)

        public_url = upload_to_gcs(filename, video_data, title, description)

        video_src = public_url
    else:
        video_src = ''

    return title, description, video_src

@app.callback(
    Output('video-player-container', 'children'),
    Input('video-url-store', 'data')
)
def update_video_player(video_url):
    if not video_url:
        raise exceptions.PreventUpdate
    
    return html.Div([
        html.Video(
            controls=True,
            src=video_url,
            style={'width': '100%'}
        )
    ])

@app.callback(
    Output('gallery-content', 'children'),
    Input('tabs', 'active_tab')
)
def update_gallery(active_tab):
    if active_tab != 'gallery-tab':
        raise exceptions.PreventUpdate

    videos = fetch_videos_metadata()
    
    if not videos:
        return html.P("No videos available.")

    gallery_items = []
    for video in videos:
        gallery_items.append(
            dbc.Card([
                html.Video(
                    controls=True,
                    src=video['url'],
                    style=styles['thumbnail']
                ),
                html.Div([
                    html.H4(video['title']),
                    html.P(video['description'])
                ], style=styles['metadata'])
            ], style=styles['card'])
        )

    return gallery_items

@app.callback(
    Output('connected-text', 'children'),
    Input('connect-button', 'n_clicks'),
    State('device-dropdown', 'value')
)
def connect_device(n_clicks, selected_device):
    if n_clicks is None:
        raise exceptions.PreventUpdate
    if selected_device:
        return f"Connected to {selected_device}"
    return ''

if __name__ == '__main__':
    app.run_server(debug=True)
