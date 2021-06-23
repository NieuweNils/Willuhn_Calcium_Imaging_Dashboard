from flask import Flask
import dash

server = Flask(__name__)
app = dash.Dash(__name__, server=server, suppress_callback_exceptions=True)
