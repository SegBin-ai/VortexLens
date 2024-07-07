from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

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

# Layout of the Dash app
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("TerraVortex", style=styles['header'])),
        dbc.Col(html.Img(src='C:\\Users\\Aaditya Voruganti\\Desktop\\VortexLens\\Dashboard\\Logo.png', style={'width': '100px', 'height': '100px'}), width=2)
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
     State('upload-video', 'contents')]
)
def update_output(n_clicks, title, description, video_content):
    if n_clicks is None:
        raise PreventUpdate

    if video_content is not None:
        video_src = video_content
    else:
        video_src = ''

    return title, description, video_src

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
