import sys

import cv2
from flask import *
from flask_cors import CORS, cross_origin

from classification.ann_model import ANN_Model
from classification.dcnn_model import DCNN_Model
from classification.diabetes import DiabetesModel
from classification.model import CategoricalMapping, Model
from classification.stroke import StrokeModel

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}})

cv2.ocl.setUseOpenCL(False)

# Create a new Machine Learning Model

# diabetes_model = DiabetesModel()
# stroke_model = StrokeModel()

# Create a new Coronary Model

coronary_name = 'coronary'
coronary_map = []
coronary_map.append(CategoricalMapping({'Present': 0, 'Absent': 1}, 'famhist'))
coronary_model = ANN_Model(coronary_name, "chd", coronary_map, [])

# Create a Diabetes Model
diabetes_name = 'diabetes'
diabetes_model = ANN_Model(diabetes_name, 'Diabetes_binary', [], [])
diabetes_features = [
        'high_bp', 'high_chol', 'chol_check', 'bmi', 'smoker', 'stroke',
        'heart_disease', 'phys_activity', 'fruits', 'veggies', 'heavy_alc',
        'health_insurance', 'no_doc_bc_cost', 'gen_health', 'mental_health',
        'phys_health', 'diff_walk', 'sex', 'age_category'
    ]

# Create a DCNN Model
dcnn_model = DCNN_Model()
# dcnn_model.predict('./test/melanocytic.jpg')
# dcnn_model.predict('./test/actinic-keratosis.jpg')
dcnn_model.predict('./test/dermatofibroma.jpg')


# Method to make a prediction route


def make_route(name, features, model):
    @app.route(f'/{name}', methods=['POST'], endpoint=name) 
    @cross_origin()
    def predict():
        data = request.get_json()
        missing_feature = [feature for feature in features if feature not in data]
        if missing_feature:
            abort(
                404, description=f'Missing required feature: {", ".join(missing_feature)}')
        convert_data = {key: [value] for key, value in data.items() if key in features}
        result = model.predict(convert_data)
        return jsonify({
            'status': f"success predict {name}",
            'result': result.tolist()
        }) 
    predict.__name__ = name
    
# Routes List
make_route(f'{coronary_name}', ['sbp','tobacco','ldl','adiposity','famhist','typea','obesity','alcohol','age'], coronary_model)
make_route(f'{diabetes_name}', diabetes_features, diabetes_model)

# @app.route('/diabetes', methods=['POST'])
# @cross_origin()
# def predict_diabetes():
#     data = request.get_json()
#     features = [
#         'high_bp', 'high_chol', 'chol_check', 'bmi', 'smoker', 'stroke',
#         'heart_disease', 'phys_activity', 'fruits', 'veggies', 'heavy_alc',
#         'health_insurance', 'no_doc_bc_cost', 'gen_health', 'mental_health',
#         'phys_health', 'diff_walk', 'sex', 'age_category'
#     ]

#     missing_feature = [feature for feature in features if feature not in data]
#     if missing_feature:
#         abort(
#             404, description=f'Missing required feature: {", ".join(missing_feature)}')

#     feature_values = [float(data[feature]) for feature in features]

#     result = diabetes_model.predict(*feature_values)

#     print(result, file=sys.stderr)

#     return jsonify({
#         'status': "success",
#         'result': result.tolist()
#     })


# @app.route('/stroke', methods=['POST'])
# @cross_origin()
# def predict_stroke():
#     data = request.get_json()

#     features = [
#         'gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status',
#     ]

#     missing_feature = [feature for feature in features if feature not in data]
#     if missing_feature:
#         abort(
#             404, description=f'Missing required feature: {", ".join(missing_feature)}')

#     feature_values = [float(data[feature]) for feature in features]

#     result = stroke_model.predict(*feature_values)

#     print(result, file=sys.stderr)

#     return jsonify({
#         'status': "success",
#         'result': result.tolist()
#     })


@app.after_request
def add_header(response):
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8000)
