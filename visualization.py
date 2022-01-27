from typing import Counter
import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt
import database as db
df = db.get_all('hotel_reviews')
df = df[['Hotel_Address',
'Additional_Number_of_Scoring',       
'Review_Date',               
'Average_Score',                               
'Hotel_Name',                                  
'Reviewer_Nationality',                        
'Negative_Review',                             
'Review_Total_Negative_Word_Counts',           
'Total_Number_of_Reviews',                     
'Positive_Review',                             
'Review_Total_Positive_Word_Counts',           
'Total_Number_of_Reviews_Reviewer_Has_Given', 
'Reviewer_Score',
'Tags', 
'days_since_review', 
'lat', 
'lng']]
data_hotels = db.get_all('data_hotels')
# df = pd.read_csv('data/Hotel_Reviews.csv')
# data_hotels = pd.read_csv('data/Data_Hotels.csv')
data = data_hotels
data = data[['Hotel_Name', 'Average_Score', 'Total_Number_of_Reviews', 'country', 'city', 'Hotel_Address','Additional_Number_of_Scoring','lat','lng']]
nationalities = pd.DataFrame.from_dict(Counter(df.Reviewer_Nationality), orient='index', columns=['country']).reset_index()
data_models = pd.read_csv('data/Data_Models.csv')

# # Wordcloud
import flask
import glob
import os

image_directory = 'images/wordclouds/'
list_of_images = [os.path.basename(x) for x in glob.glob('{}*.png'.format(image_directory))]
static_image_route = '/static/'
# def display_wordcloud(df):
#     my_stopwords = ENGLISH_STOP_WORDS.union(['hotel', 'room', 'staff', 'rooms', 'breakfast', 'bathroom'])

#     my_cloud = WordCloud(background_color='white',stopwords=my_stopwords).generate(' '.join(df['review']))
#     plt.imshow(my_cloud, interpolation='bilinear') 
#     plt.axis("off")
#     plt.show()

#     df_positive = df[df['label']==1]
#     my_cloud_positive = WordCloud(background_color='white',stopwords=my_stopwords).generate(' '.join(df_positive['review']))
#     plt.imshow(my_cloud_positive, interpolation='bilinear') 
#     plt.axis("off")
#     plt.show()

#     df_negative = df[df['label']==0]
#     my_cloud_negative = WordCloud(background_color='white',stopwords=my_stopwords).generate(' '.join(df_negative['review']))
#     plt.imshow(my_cloud_negative, interpolation='bilinear') 
#     plt.axis("off")
#     plt.show()

# display_wordcloud(data_models)

# Table
table = go.Figure(data=[go.Table(
    header=dict(values=['Hotel name','Hotel address','Additional number of scoring','Average score','Total number of reviews','Country','City'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[data.Hotel_Name, data.Hotel_Address, data.Additional_Number_of_Scoring, data.Average_Score, data.Total_Number_of_Reviews, data.country, data.city],
               fill_color='lavender',
               align='left'))
])

# Mapa
mapa = px.scatter_mapbox(data_hotels, lat="lat", lon="lng", color="Average_Score", size="Total_Number_of_Reviews",
                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10, hover_name=data_hotels.Hotel_Name)

mapa.update_layout(
    mapbox_style="open-street-map",
    title='Hotels ratings map',
    autosize=True,
    hovermode='closest',
    showlegend=False,
)

# Bubble graph
# Applying aggregation function in MongoDB (instead of query):
data_triple = db.aggregate_fun()

triple = px.scatter(data_triple, x="Additional_Number_of_Scoring", y="Average_Score", size="Total_Number_of_Reviews", color="Hotel_Name",
           hover_name="Hotel_Name", size_max=60, width=920, title = 'Hotels scores and number of reviews')

# Pie chart
pie = px.pie(nationalities.query("country >= 5000"), values='country', names='index', title='Reviewers nationalities')


logo_link = 'https://zakelijkschrijven.nl/wp-content/uploads/2021/01/HvA-logo.png'
  
data_top_num_reviews = data.nlargest(10, 'Total_Number_of_Reviews')
fig1 = px.bar(data_top_num_reviews, x='Hotel_Name', y='Total_Number_of_Reviews', color='Hotel_Name')                 
list_names = ['Name', 'Average score', 'Number of reviews', 'Country', 'City', 'Adress', 'Additional score', 'lat', 'long']
cols=[
            {"name": i, "id": i, "deletable": True, "selectable": True} for i in data.columns
        ]
cols[0]["name"] = "Name"
cols[1]["name"] = "Average score"
cols[2]["name"] = "Number of reviews"
cols[3]["name"] = "Country"
cols[4]["name"] = "City"
cols[5]["name"] = "Adress"
cols[6]["name"] = "Additional score"
cols[7]["name"] = "lat"
cols[8]["name"] = "long"

opts =[{'label': i, 'value': i} for i in list_of_images]
opts[0]["label"] = "Negative Wordcloud"
opts[1]["label"] = "Positive Wordcloud"
opts[2]["label"] = "Total Wordcloud"

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app = dash.Dash(__name__)

app.layout = html.Div([
  html.Img(src=logo_link, width="500", height="250"),
  # Add margin to the logo
  html.H1('Hotels Dashboard'),

  html.Div(style={
                    'height':'100px',
                    'margin-left':'10px',
                    'width':'80%',
                    'text-align':'center',
                    'display':'inline-block'},children=[
    dash_table.DataTable(
        id='datatable-interactivity',
        columns = cols,
        data=data.to_dict('records'),
        style_table={'overflowX': 'auto'},
        editable=True,
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        column_selectable="single",
        row_selectable="multi",
        row_deletable=True,
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        page_current= 0,
        page_size= 10,
        style_cell={
        'height': 'auto',
        # all three widths are needed
        'minWidth': '180px', 'width': '90px', 'maxWidth': '90px',
         'textAlign': 'left'},
        style_cell_conditional=[
            {
                'if': {'column_id': 'Hotel_Name'},
                'textAlign': 'left'
            }
        ]
    ),
    html.Div(id='datatable-interactivity-container', style={'display':'inline-block'})
    ]),
################# Select hotel in the map and get information below
  html.Div([
      html.Div(style={
                    'height':'100px',
                    'margin-left':'10px',
                    'width':'45%',
                    'text-align':'center',
                    'display':'inline-block'},
                children=dcc.Graph(
                    id='example-map',
                    figure=mapa,
                    style={'display':'inline-block'}
          
          )),
        html.Div(style={
                    'height':'100px',
                    'margin-left':'10px',
                    'width':'45%',
                    'text-align':'center',
                    'display':'inline-block'},
            children=[dcc.Dropdown(
                id='image-dropdown',
                options=opts,
                value=list_of_images[2]),
            html.Img(id='image')])
    ]),
  
  html.Div(children=[
      dcc.Graph(
          id='triple-graph',
          figure=triple,
          style={'display':'inline-block'}),
      dcc.Graph(
        # Style the graphs to appear side-by-side
        figure=pie,
        style={'display':'inline-block'})
  ]),

  ], style={'text-align':'center', 'font-size':22})

####### Filters
@app.callback(
    Output('datatable-interactivity', 'style_data_conditional'),
    Input('datatable-interactivity', 'selected_columns')
)
def update_styles(selected_columns):
    return [{
        'if': { 'column_id': i },
        'background_color': '#D2F3FF'
    } for i in selected_columns]

@app.callback(
    Output('datatable-interactivity-container', "children"),
    Input('datatable-interactivity', "derived_virtual_data"),
    Input('datatable-interactivity', "derived_virtual_selected_rows"))
def update_graphs(rows, derived_virtual_selected_rows):
    # When the table is first rendered, `derived_virtual_data` and
    # `derived_virtual_selected_rows` will be `None`. This is due to an
    # idiosyncrasy in Dash (unsupplied properties are always None and Dash
    # calls the dependent callbacks when the component is first rendered).
    # So, if `rows` is `None`, then the component was just rendered
    # and its value will be the same as the component's dataframe.
    # Instead of setting `None` in here, you could also set
    # `derived_virtual_data=df.to_rows('dict')` when you initialize
    # the component.
    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows = []

    dff = data if rows is None else pd.DataFrame(rows)

    colors = ['#7FDBFF' if i in derived_virtual_selected_rows else '#0074D9'
              for i in range(len(dff))]

    return [
        # dcc.Graph(
        #     id=column,
        #     figure={
        #         "data": [
        #             {
        #                 "x": dff["Hotel_Name"],
        #                 "y": dff[column],
        #                 "type": "bar",
        #                 "marker": {"color": colors},
        #             }
        #         ],
        #         "layout": {
        #             "xaxis": {"automargin": True},
        #             "yaxis": {
        #                 "automargin": True,
        #                 "title": {"text": column}
        #             },
        #             "height": 250,
        #             "margin": {"t": 10, "l": 10, "r": 10},
        #         },
        #     },
        # )
        # # check if column exists - user may have deleted it
        # # If `column.deletable=False`, then you don't
        # # need to do this check.
        # for column in ['Hotel_Address', 'Additional_Number_of_Scoring', 'Average_Score', 'Total_Number_of_Reviews', 'country', 'city'] if column in dff
    ]

@app.callback(
    dash.dependencies.Output('image', 'src'),
    [dash.dependencies.Input('image-dropdown', 'value')])
def update_image_src(value):
    return static_image_route + value

# Add a static image route that serves images from desktop
# Be *very* careful here - you don't want to serve arbitrary files
# from your computer or server
@app.server.route('{}<image_path>.png'.format(static_image_route))
def serve_image(image_path):
    image_name = '{}.png'.format(image_path)
    if image_name not in list_of_images:
        raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
    return flask.send_from_directory(image_directory, image_name)

if __name__ == '__main__':
    app.run_server(debug=True)