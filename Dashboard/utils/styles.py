secondary_color = '#FFFFFF'  # White
accent_color = '#DC143C'  # Navy Blue
primary_color = '#FFFFFF'  # White
styles = {
    'container': {
        'backgroundColor': primary_color,
        'color': accent_color,
        'padding': '20px'
    },
    'header': {
        'textAlign': 'center',
        'color': accent_color,
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
        'borderColor': accent_color,
        'color': accent_color,
        'marginTop': '20px'
    },
    'video': {
        'width': '100%',
        'marginTop': '20px'
    },
    'dropdown': {'width': '100%', 'marginBottom': '10px'},  # Added this line

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
    },
    'connected-text': {
        'color': accent_color,
        'fontSize': '20px',
        'marginTop': '10px'
    }
}