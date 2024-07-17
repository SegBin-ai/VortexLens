from dash import dcc, html, Input, Output, State, exceptions
import dash_bootstrap_components as dbc
import base64
from utils.styles import styles
from utils.google_cloud import upload_to_gcs, get_all_structures

# Fetch initial list of structures
initial_structures = get_all_structures()

# Layout for the Upload page
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
            dbc.Label("Select Structure"),
            dcc.Dropdown(
                id='structure-dropdown',
                options=[{'label': s, 'value': s} for s in initial_structures] + [{'label': 'Create New', 'value': 'create-new'}],
                value='create-new'
            ),
            html.Div(id='new-structure-container', children=[
                dbc.Label("New Structure Name"),
                dbc.Input(id='new-structure-name', type='text', placeholder='Enter new structure name', style=styles['input']),
                dbc.Button('Add Structure', id='add-structure-button', style=styles['button'])
            ], style={'display': 'none'}),
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

def register_upload_callbacks(app):
    @app.callback(
        Output('output-filename', 'children'),
        Input('upload-video', 'filename')
    )
    def update_filename(filename):
        if filename is not None:
            return f"Uploaded file: {filename}"
        return 'No file uploaded.'

    @app.callback(
        [Output('output-title', 'children'),
         Output('output-description', 'children'),
         Output('video-url-store', 'data')],
        [Input('submit-button', 'n_clicks')],
        [State('video-title', 'value'),
         State('video-description', 'value'),
         State('upload-video', 'contents'),
         State('upload-video', 'filename'),
         State('structure-dropdown', 'value')]
    )
    def update_output(n_clicks, title, description, video_content, filename, selected_structure):
        if n_clicks is None:
            raise exceptions.PreventUpdate

        if video_content is not None:
            content_type, content_string = video_content.split(',')
            video_data = base64.b64decode(content_string)

            # Upload to GCS and get public URL
            public_url = upload_to_gcs(filename, video_data, title, description, selected_structure)

            # Debugging output
            print(f"Uploaded video URL: {public_url}")

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
        
        # Debugging output
        print(f"Video URL for player: {video_url}")

        return html.Div([
            html.Video(
                controls=True,
                src=video_url,
                style={'width': '100%'}
            )
        ])

    @app.callback(
        [Output('structure-dropdown', 'options'),
         Output('structure-dropdown', 'value'),
         Output('new-structure-container', 'style'),
         Output('structure-list-store', 'data')],
        [Input('add-structure-button', 'n_clicks'),
         Input('structure-dropdown', 'value')],
        [State('new-structure-name', 'value'),
         State('structure-list-store', 'data')]
    )
    def update_structures(n_clicks, selected_value, new_structure_name, current_structures):
        if n_clicks and new_structure_name:
            if new_structure_name not in current_structures:
                current_structures.append(new_structure_name)
            return [{'label': s, 'value': s} for s in current_structures] + [{'label': 'Create New', 'value': 'create-new'}], new_structure_name, {'display': 'none'}, current_structures
        
        if selected_value == 'create-new':
            return [{'label': s, 'value': s} for s in current_structures] + [{'label': 'Create New', 'value': 'create-new'}], 'create-new', {'display': 'block'}, current_structures
        
        return [{'label': s, 'value': s} for s in current_structures] + [{'label': 'Create New', 'value': 'create-new'}], selected_value, {'display': 'none'}, current_structures
