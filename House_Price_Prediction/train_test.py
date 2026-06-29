import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest,mutual_info_regression
from sklearn.model_selection import train_test_split
import os

try :
    from category_encoders import TargetEncoder
except ImportError:
    TargetEncoder = None
    print("Warning: category_encoders not installed.Target Encoding will be unavaliable")
finally:
    print("It's Installed")

def main():
    print("Loading Dataset:")
    file_path = "dataset.csv"

    if not os.path.exists(file_path):
        print(f"Error cannot find '{file_path}'")
        return

    df = pd.read_csv(file_path)
    print(f"Dataset Loaded Successfully.\n Rows : {df.shape[0]},Features:{df.shape[1]}\n")

    print(df.isnull().sum())
    df['ID'] = ["ID_"+str(np.random.randint(1,150))for _ in range(len(df))]
    
    if TargetEncoder is not None:
        print("Applying Target Encoder")

        encoder = TargetEncoder()

        df['ID_Encoded'] = encoder.fit_transform(df['ID'],df['price'])
        print(df.head())

    else:
        print("Category Encoders not installed")

    #feature Selection

    features_to_test = ["price","bedrooms","sqft_living","sqft_lot","floors","condition"]
    X_features = df[features_to_test].fillna(0)
    y_target = df['price']

    selector = SelectKBest(score_func=mutual_info_regression,k=4)
    selector.fit(X_features,y_target)
    winning_features = selector.get_support()
    best_features = X_features.columns[winning_features].tolist()

    print(best_features)

    # Splitting Data
    X =df[best_features]
    Y =df['price']

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
    print(df.shape)
    print(f"Training Data Size: {X_train.shape}")
    print(f"Testing Data Size: {X_test.shape}")

    # Training Model

    model = LinearRegression()
    model.fit(X_train,Y_train)

    predictions = model.predict(X_test)

    print(np.round(predictions)) 
    print(predictions.shape)

    # compare model predictions to the actual answer

    actual_wins = Y_test.head(3).values
    predicted_wins = predictions[:3]

    for i in range(3):
        predicted = round(predicted_wins[i])
        actual = actual_wins[i]
        difference = abs(actual-predicted)

        print(f"Model Guessed:{predicted}")
        print(f"Real Answer:{actual}")
        print(f"Difference:{difference}")

        
if __name__ == "__main__":
    main()