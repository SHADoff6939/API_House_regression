import pandas as pd
import json


class DataLoader(object):
    def fit(self, dataset):
        self.dataset = dataset.copy()
        with open('settings/specifications.json') as f:
            specifications = json.load(f)
        self.final_columns = specifications['description']['final_columns']

    def load_data(self, is_train=False):
        df = self.dataset

        # 1 drop cols
        cols_to_del = ['Id', 'LotFrontage', 'Alley', 'MasVnrType', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
        df.drop(cols_to_del, inplace=True, axis=1)


        # 2 fill NA in MasVnrArea w mean
        df['MasVnrArea'] = df.groupby('MSSubClass')['MasVnrArea'].transform(lambda x: x.fillna(x.mean()))

        # 3 same with mode in Electrical
        df['Electrical'] = df.groupby('MSSubClass')['Electrical'].transform(lambda x: x.fillna(x.mode()[0]))

        # 4 Fill NA values in basement cols with 0 if it all has NA
        cat_bsmt_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

        df.loc[df[cat_bsmt_cols].isnull().all(axis=1), cat_bsmt_cols] = 'No Basement'

        # 5 fill NA same way but in garage col
        cat_garage_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']

        df.loc[df[cat_garage_cols].isnull().all(axis=1), cat_garage_cols] = 'No Garage'

        # Fill NA in GarageYrBlt w 0
        df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)

        # All data is filled now it's time to prep it

        # Lists with columns for further work with them
        one_code_cols = ['MSZoning', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2',
                         'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
                         'Exterior2nd', 'Foundation', 'Heating', 'CentralAir', 'GarageType',
                         'SaleType']

        rating_cols = ['Street', 'LotShape', 'LandContour', 'Utilities', 'LandSlope',
                       'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                       'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'Electrical', 'KitchenQual',
                       'Functional', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',
                       'SaleCondition']

        cat_cols = [
            'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
            'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
            'HouseStyle', 'OverallQual', 'OverallCond', 'RoofStyle', 'RoofMatl', 'Exterior1st',
            'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
            'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC',
            'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType',
            'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence',
            'MiscFeature', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition'
        ]
        cont_cols = [
            'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
            'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
            'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
            'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea',
            'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
            'PoolArea', 'MiscVal', 'YearBuilt', 'YearRemodAdd'
        ]

        # 7 one-hot encoding for list of columns
        existing_cols = [col for col in one_code_cols if col in df.columns]
        df = pd.get_dummies(df, columns=existing_cols, drop_first=True)

        # 8 Label Encoding
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()

        for col in rating_cols:
            df[col] = le.fit_transform(df[col])

        # 9 transform data with standart scaling
        from sklearn.preprocessing import StandardScaler

        std_scaler = StandardScaler()
        for col in cont_cols:
            if col in df.columns:
                df[col] = std_scaler.fit_transform(df[[col]])

        # 10 handle outliers with replacing them with values
        weak_outliers_cols = ['OverallCond', 'LotArea', 'OpenPorchSF', 'MasVnrArea']

        for column in weak_outliers_cols:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)

            IQR = Q3 - Q1

            lower_limit = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR

            df.loc[df[column] < lower_limit, column] = lower_limit
            df.loc[df[column] > upper_limit, column] = upper_limit

        # match columns with final list

        if is_train:
            self.final_columns = df.columns.tolist()
        else:
            # Для predict: якщо колонки відсутні — додаємо їх з нулями
            for col in self.final_columns:
                if col not in df.columns:
                    df[col] = 0

            # Якщо зайві — відкидаємо
            df = df[self.final_columns]


        return df