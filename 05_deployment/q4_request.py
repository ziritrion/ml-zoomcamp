import requests

url = 'http://192.168.64.2:9696/predict'
customer = {"contract": "two_year", "tenure": 1, "monthlycharges": 10}
response = requests.post(url, json=customer).json()
print(f'The churn probability will be {response["churn_probability"]}')
if response['churn']==True:
    print('Customer will churn! sending promo email to customer.')
else:
    print('No churn lol')