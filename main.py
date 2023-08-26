import sys

import cv2
from flask import *
from flask_cors import CORS, cross_origin

from classification.ann_model import ANN_Model
from classification.dcnn_model import DCNN_Model
# from classification.dcnn_model import DCNN_Model
from classification.diabetes import DiabetesModel
from classification.model import CategoricalMapping, Model
from classification.stroke import StrokeModel

app = Flask(__name__)
CORS(app)

cv2.ocl.setUseOpenCL(False)

# Create a new Machine Learning Model

# diabetes_model = DiabetesModel()  
# stroke_model = StrokeModel()

# Create a new Coronary Model

coronary_name = 'coronary'
coronary_categorical = []
coronary_categorical.append(CategoricalMapping({'Present': 0, 'Absent': 1}, 'famhist'))
coronary_result = "chd"
coronary_drop_list = []
coronary_ann_model = ANN_Model(coronary_name, coronary_result, coronary_categorical, coronary_drop_list)
coronary_model = Model(coronary_name, coronary_result, coronary_categorical, coronary_drop_list)
coronary_feature = ['sbp', 'tobacco', 'ldl', 'adiposity',
           'famhist', 'typea', 'obesity', 'alcohol', 'age']

# Create a Diabetes Model
diabetes_name = 'diabetes'
diabetes_result = 'Diabetes_binary'
diabetes_categorical = []
diabetes_drop_list = ['Education', 'Income']
diabetes_ann_model = ANN_Model(diabetes_name, diabetes_result, diabetes_categorical, diabetes_drop_list)
diabetes_model = Model(diabetes_name, diabetes_result, diabetes_categorical, diabetes_drop_list)
diabetes_features = [
    'HighBP',
    'HighChol',
    'CholCheck',
    'BMI',
    'Smoker',
    'Stroke',
    'HeartDiseaseorAttack',
    'PhysActivity',
    'Fruits',
    'Veggies',
    'HvyAlcoholConsump',
    'AnyHealthcare',
    'NoDocbcCost',
    'GenHlth',
    'MentHlth',
    'PhysHlth',
    'DiffWalk',
    'Sex',
    'Age'
]

# Create a Stroke Model
stroke_name = 'stroke'
stroke_categorical = [
    CategoricalMapping({'Male': 0, 'Female': 1, 'Other': 2}, 'gender'),
    CategoricalMapping({'No': 0, 'Yes': 1}, 'ever_married'),
    CategoricalMapping({'children': 0, 'Govt_job': 1, 'Never_worked': 2, 'Private': 3 , 'Self-employed': 4}, 'work_type'),
    CategoricalMapping({'Rural': 0, 'Urban': 1}, 'residence_type'),
    CategoricalMapping({'formerly smoked': 3, 'never smoked': 2, 'smokes': 1, 'Unknown': 0}, 'smoking_status'),
    ]
stroke_result = 'stroke'
stroke_drop_list = ['id']
stroke_ann_model = ANN_Model(stroke_name, stroke_result, stroke_categorical, stroke_drop_list)
stroke_model = Model(stroke_name, stroke_result, stroke_categorical, stroke_drop_list)

stroke_features = [
'gender','age','hypertension','heart_disease','ever_married','work_type','residence_type','avg_glucose_level','bmi','smoking_status']

# Create a Mental Model
mental_name  = 'mental'
mental_categorical = [
    CategoricalMapping({'Male': 0, 'Female': 1}, 'Gender')
    # CategoricalMapping({'0': 0, '100-500': 1, '26-100': 2, '500-1000': 3,'More than 1000': 4}, 'no_employees'),
]
mental_drop_list = ['Timestamp', 'Country' , 'state', 'no_employees']
mental_result = 'treatment'
mental_ann_model = ANN_Model(mental_name, mental_result, mental_categorical , mental_drop_list)
mental_model = Model(mental_name, mental_result, mental_categorical , mental_drop_list)

mental_features = ['Age','Gender','self_employed','family_history','work_interfere','remote_work','tech_company','benefits','care_options','wellness_program','seek_help','anonymity','leave','mental_health_consequence','phys_health_consequence','coworkers','supervisor','mental_health_interview','phys_health_interview','mental_vs_physical','obs_consequence']

# Create a Heart Model
cardio_name = 'cardio'
cardio_drop_list = ['id']
cardio_categorical_list = []
cardio_result = 'cardio'
cardio_ann_model = ANN_Model(cardio_name, cardio_result, cardio_categorical_list, cardio_drop_list)
cardio_model = Model(cardio_name, cardio_result, cardio_categorical_list, cardio_drop_list)
cardio_feature = ['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']

# Create a DCNN Model
dcnn_model = DCNN_Model()
# dcnn_model.predict('./test/melanocytic.jpg')
# dcnn_model.predict('./test/actinic-keratosis.jpg')
# dcnn_model.predict('./test/dermatofibroma.jpg')

# Method to make a prediction route

def make_route(name, features, model):
    @app.route(f'/{name}', methods=['POST'], endpoint=name)
    @cross_origin()
    def predict():
        data = request.get_json()
        missing_feature = [
            feature for feature in features if feature not in data]
        if missing_feature:
            abort(
                404, description=f'Missing required feature: {", ".join(missing_feature)}')
        convert_data = {key: [value]
                        for key, value in data.items() if key in features}
        result = model.predict(convert_data)
        return jsonify({
            'status': f"success predict {name}",
            'result': result.tolist()[0]
        })
    predict.__name__ = name

# Routes List

# Default Model Route List
make_route(f'{coronary_name}', coronary_feature, coronary_model)
make_route(f'{diabetes_name}', diabetes_features, diabetes_model)
make_route(f'{stroke_name}', stroke_features, stroke_model)
make_route(f'{mental_name}', mental_features, mental_model)
make_route(f'{cardio_name}', cardio_feature, cardio_model)

# ANN Model Route List
make_route(f'{coronary_name}-ann', coronary_feature, coronary_ann_model)
make_route(f'{diabetes_name}-ann', diabetes_features, diabetes_ann_model)
make_route(f'{stroke_name}-ann', stroke_features, stroke_ann_model)
make_route(f'{mental_name}-ann', mental_features, mental_ann_model)
make_route(f'{cardio_name}-ann', cardio_feature, cardio_ann_model)

# DCNN Model Routes

@app.route(f'/dcnn', methods=['POST'])
@cross_origin()
def predict_dcnn():
    data = request.get_json()
    if(data['link'] is None):
        abort(
            404, description=f'Missing link')

    result = dcnn_model.predict(data['link'])
    return jsonify({
        'status': f"success predict dcnn",
        'result': result,
    })

# Test Routes
@app.route('/', methods=['GET'])
@cross_origin()
def test():
    return 'Success'

@app.after_request
def add_header(response):
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8000)
