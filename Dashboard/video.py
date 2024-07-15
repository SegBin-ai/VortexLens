from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import base64
import os
from google.cloud import storage

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define styles
secondary_color = '#DC143C'  # Crimson
accent_color = '#000080'  # Navy Blue
primary_color = '#FFFFFF'  # White

styles = {
    'container': {
        'backgroundColor': primary_color,
        'color': accent_color,
        'padding': '20px'
    },
    'header': {
        'textAlign': 'center',
        'color': secondary_color,
    },
    'upload': {
        'width': '100%',
        'height': '60px',
        'lineHeight': '60px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center',
        'margin': '10px',
        'backgroundColor': secondary_color,
        'color': accent_color,
    },
    'button': {
        'backgroundColor': secondary_color,
        'borderColor': secondary_color,
        'color': accent_color,
        'marginTop': '20px'
    },
    'video': {
        'width': '100%',
        'marginTop': '20px'
    },
    'input': {
        'backgroundColor': secondary_color,
        'color': accent_color,
        'borderColor': secondary_color,
        'marginBottom': '10px'
    },
    'textarea': {
        'backgroundColor': secondary_color,
        'color': accent_color,
        'borderColor': secondary_color,
        'marginBottom': '10px'
    },
    'gallery': {
        'display': 'flex',
        'flexWrap': 'wrap',
        'justifyContent': 'space-around',
        'paddingTop': '20px'
    },
    'card': {
        'width': '300px',
        'margin': '10px',
        'border': f'1px solid {secondary_color}',
        'borderRadius': '5px',
        'boxShadow': '2px 2px 5px rgba(0,0,0,0.1)'
    },
    'thumbnail': {
        'width': '100%'
    },
    'metadata': {
        'padding': '10px'
    }
}

# Google Cloud Storage setup
bucket_name = 'factory-work'  # Replace with your bucket name

def upload_to_gcs(file_name, file_content, title, description):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    
    metadata = {
        'title': title,
        'description': description
    }
    
    blob.metadata = metadata
    blob.upload_from_string(file_content, content_type='video/mp4')
    return blob.public_url

def fetch_videos_metadata():
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs()

    videos = []
    for blob in blobs:
        if blob.metadata and 'title' in blob.metadata and 'description' in blob.metadata:
            video_info = {
                'url': blob.public_url,
                'title': blob.metadata['title'],
                'description': blob.metadata['description']
            }
            videos.append(video_info)
    return videos

# Layout of the Dash app
app.layout = dbc.Container([
    dbc.Tabs([
        dbc.Tab(label='Upload', tab_id='upload-tab'),
        dbc.Tab(label='Gallery', tab_id='gallery-tab')
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
                multiple=False
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
            html.Video(id='output-video', controls=True, style=styles['video'])
        ])
    ])
])

gallery_layout = html.Div([
    html.H1("Video Gallery", style=styles['header']),
    html.Div(id='gallery-content', style=styles['gallery'])
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
     Output('output-video', 'src')],
    [Input('submit-button', 'n_clicks')],
    [State('video-title', 'value'),
     State('video-description', 'value'),
     State('upload-video', 'contents'),
     State('upload-video', 'filename')]
)
def update_output(n_clicks, title, description, video_content, filename):
    if n_clicks is None:
        raise PreventUpdate

    if video_content is not None:
        # Decode the base64 video content
        content_type, content_string = video_content.split(',')
        video_data = base64.b64decode(content_string)

        # Upload to Google Cloud Storage
        public_url = upload_to_gcs(filename, video_data, title, description)

        # Return the public URL to display the video
        video_src = public_url
    else:
        video_src = ''

    return title, description, video_src

@app.callback(
    Output('gallery-content', 'children'),
    Input('tabs', 'active_tab')
)
def update_gallery(active_tab):
    if active_tab != 'gallery-tab':
        raise PreventUpdate

    videos = fetch_videos_metadata()
    
    if not videos:
        return html.P("No videos available.")

    gallery_items = []
    for video in videos:
        gallery_items.append(
            dbc.Card([
                html.Video(src=video['url'], controls=True, style=styles['thumbnail']),
                html.Div([
                    html.H4(video['title']),
                    html.P(video['description'])
                ], style=styles['metadata'])
            ], style=styles['card'])
        )

    return gallery_items

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
