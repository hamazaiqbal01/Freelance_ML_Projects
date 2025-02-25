# import sys
# from dataclasses import dataclass 
# import os
# import numpy as np 
# import pandas as pd 

# from src.exception import CustomException

# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from src.logger import logging 

# from src.utils import save_object # it used for savig pkl file 

# @dataclass 
# class DataTransformationConfig:
#     preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

# class DataTransformation:
#     def __init__(self):
#         self.data_transformation_config = DataTransformationConfig()

#     def get_data_transformer_object(sefl):

#         '''
#         This function is responsible for data transformation based on different datasets 
#         '''
#         try:
#             numerical_coloumns = ["writing_score","reading_score"]
#             categorical_coloumns = [
#                 "gender"
#                 "race_ethnicity"
#                 "parental_level_of_education"
#                 "lunch"
#                 "test_prepration_course",
#             ]
#             num_pipeline = Pipeline(
#                 steps= [
#                     ("imputer",SimpleImputer(stratergy="median"))
#                     ('scalar',StandardScaler)
#                 ]

#             cat_pipeline= Pipeline(
#                 steps=[
#                     ('imputer',SimpleImputer(stratergy="most_frequent"))
#                     ("one_hot_encoder",OneHotEncoder())
#                     ('Scalar',StandardScaler())

#                     ]
#                 )
#             )
#             logging.info("Numerical columns standard scaling completed ")
#             logging.info("Categorical columns encoding completed ")

#             preprocessor = ColumnTransformer(
#                 [
#                     ("num_pipeline",  num_pipeline,numerical_coloumns)
#                     ("cat_pipelines", cat_pipeline, categorical_coloumns)
#                 ]
#             )
#             return preprocessor
#         except Exception as e:
#             raise CustomException(e,sys)
    
#     def initiate_data_transformation(self,train_path,test_path):
#         try:
#             train_df = pd.read_csv(train_path)
#             test_df = pd.read_csv(test_path)

#             logging.info("Read train and test data completed ")
#             logging.info("Obtaining Preprocessing object ")

#             preprocessing_obj= self.get_data_transformer_object()
#             target_column_name= " math_score"
#             numerical_coloumns = ["writing_score","reading_score"]

#             input_feature_train_df= train_df.drop(columns=[target_column_name])

#         except:
#             pass 




import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            print(f"Saving preprocessor object to: {self.data_transformation_config.preprocessor_obj_file_path}")  # Debugging

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)