from dash import dcc, html, Input, Output, exceptions
import dash_bootstrap_components as dbc
from utils.styles import styles
from utils.google_cloud import fetch_videos_metadata

# Layout for the Gallery page
gallery_layout = html.Div([
    html.H1("Video Gallery", style=styles['header']),
    html.Div(id='gallery-content', style=styles['gallery'])
])

def register_gallery_callbacks(app):
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
        for index, video in enumerate(videos):
            video_id = video.get('id', index)  # Use 'id' if available, otherwise use index as id
            gallery_items.append(
                dbc.Card([
                    html.Video(
                        controls=True,
                        src=video['url'],
                        style=styles['thumbnail']
                    ),
                    html.Div([
                        html.H4(video['title']),
                        html.P(video['description']),
                        html.P(video['structure']),
                        dbc.Button('Upload', id=f'upload-button-{video_id}', style=styles['button'])
                    ], style=styles['metadata'])
                ], style=styles['card'])
            )

        return gallery_items
