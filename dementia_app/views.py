from django.shortcuts import render
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def predict_result(request):
    if request.method == 'POST':
        try:
            mr = float(request.POST.get('mr'))
            age = float(request.POST.get('age'))
            educ = float(request.POST.get('educ'))
            ses = float(request.POST.get('ses'))
            mmse = float(request.POST.get('mmse'))
            cdr = float(request.POST.get('cdr'))
            etiv = float(request.POST.get('etiv'))
            nwbv = float(request.POST.get('nwbv'))
            asf = float(request.POST.get('asf'))
        except (TypeError, ValueError):
            return render(request, 'user_input.html', {'error': 'Invalid input values'})

        try:
            data = pd.read_csv('C:/Users/Lenovo/Desktop/FINAL_INTERNSHIP/DEMENTIA/dementia_project/demented_data.csv')
            
            required_columns = ['Group', 'SES', 'MMSE', 'MR Delay', 'Age', 'EDUC', 'CDR', 'eTIV', 'nWBV', 'ASF']
            if not all(col in data.columns for col in required_columns):
                raise ValueError('One or more required columns are missing in the dataset.')

            le_gr = LabelEncoder()
            data['new_gr'] = le_gr.fit_transform(data['Group'])

            median_ses = data['SES'].median()
            data['SES'] = data['SES'].fillna(median_ses)

            median_mmse = data['MMSE'].median()
            data['MMSE'] = data['MMSE'].fillna(median_mmse)

            inputs = data[['MR Delay', 'Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']]
            output = data['Group']

            model = DecisionTreeClassifier()
            model.fit(inputs, output)

            prediction = model.predict([[mr, age, educ, ses, mmse, cdr, etiv, nwbv, asf]])

            return render(request, 'user_input.html', {'prediction': prediction[0]})
        except Exception as e:
            return render(request, 'user_input.html', {'error': f'Error: {str(e)}'})
    else:
        return render(request, 'user_input.html')
