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
    }
}

# Google Cloud Storage setup
bucket_name = 'factory-work'  # Replace with your bucket name

def upload_to_gcs(file_name, file_content):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    if file_name.endswith('.mp4'):
        content_type = 'video/mp4'
    elif file_name.endswith('.mov'):
        content_type = 'video/quicktime'
    else:
        raise ValueError("Unsupported file type")

    blob.upload_from_string(file_content, content_type=content_type)
    return blob.public_url

# Layout of the Dash app
app.layout = dbc.Container([
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
            html.Video(id='output-video', controls=True, style=styles['video'])
        ])
    ])
], fluid=True, style=styles['container'])

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
        public_url = upload_to_gcs(filename, video_data)

        # Return the public URL to display the video
        video_src = public_url
    else:
        video_src = ''

    return title, description, video_src

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
