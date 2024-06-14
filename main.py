#flask, scikit-learn, pandas, pickle-mixin
# import pandas as pd
# from flask import render_template, request

# app = Flask(__name__)
# data=pd.read_csv('Cleaned_data.csv')

# @app.route('/')
# def index():
#     locations = sorted(data['location'].unique())
#     return render_template('index.html')

# if __name__ == "__main__":
#     app.run(debug=False, port=5000)




# from flask import Flask
# from gevent.pywsgi import WSGIServer

# app = Flask(__name__)

# @app.route('/api', methods=['GET'])
# def index():
#     return "Hello, World!"

# if __name__ == '__main__':
#     # Debug/Development
#     # app.run(debug=True, host="0.0.0.0", port="5000")
#     # Production
#     http_server = WSGIServer(('', 5000), app)
#     http_server.serve_forever()



# from flask import Flask

# app = Flask(__name__)

# @app.route("/")
# def index():
#     return "<h1>Hello!</h1>"

# def create_app():
#     return app
import pandas as pd
from flask import Flask, render_template, request
from waitress import serve
import pickle
import numpy as np

app = Flask (__name__)
data=pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("RidgeModel.pkl",'rb'))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template("index.html", locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')
    print(location,bhk,bath,sqft)

    # return ""
    input = pd.DataFrame([[location,sqft,bath,bhk]],columns=['location','total_sqft','bath','bhk'])

    # # Convert 'baths' column to numeric with errors='coerce'
    # input['bath'] = pd.to_numeric(input['bath'], errors='coerce')

    # Convert input data to numeric types
    input = input.astype({'bhk': float, 'bath': float})

    # # Handle unknown categories in the input data
    # for location in input.location:
    #     unknown_categories = set(input[location]) - set(data[location].unique())
    #     if unknown_categories:
    #         print(f"Unknown categories in {location}: {unknown_categories}")
    #         # Handle unknown categories (e.g., replace with a default value)
    #         input[location] = input[location].replace(unknown_categories, data[location].mode()[0])

    # print("Processed Input Data:")
    # print(input)

    prediction = pipe.predict(input)[0]
    return str(np.round(prediction,2))

if __name__=="__main__":
    # app.run()
    serve (app, host='0.0.0.0', port=50100, threads=2)
    # http_server = WSGIServer(('', 5000), app)
    # http_server.serve_forever()