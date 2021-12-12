import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


def pred(data):
    data = np.array(data)

    # Load Data

    def create_df(da, year=[]):
        data = da.copy()
        nd = pd.DataFrame()
        data.drop(["Provinsi","Stasiun BMKG"], axis=1, inplace=True)
        length = int(data.shape[1]/5)
        col_names = data.columns[0:length]
        for i in range(0,5):
            val  = data.iloc[:,0:length]
            d = dict(zip(val.columns[0::1],  col_names))
            val = val.rename(columns=d)
            if(len(year)>0):
                val["year"] = year[i]
            nd = nd.append(val, ignore_index=True)
            data.drop(data.columns[0:length], axis=1, inplace=True)
        return nd

    da1 = create_df(pd.read_excel("data/Indo_151_15887472.xls"))
    da2 = create_df(pd.read_excel("data/Indo_151_17952722.xls"))
    da3 = create_df(pd.read_excel("data/Indo_151_18467275.xls"))
    da4 = create_df(pd.read_excel("data/Indo_151_21445257.xls"),  year = [2011,2012,2013,2014,2015])

    da_list = [da1,da2,da3,da4]
    data_cuaca = pd.concat(da_list, axis=1, join="inner")
    col = ['tekanan_udara', 'penyinaran_matahari', 'suhu_minimum', 'suhu_rata_rata', 'suhu_maksimum', 'curah_hujan', 'hari_hujan', 'kecepatan_angin', 'kelembaban', 'tahun']
    data_cuaca.columns = col
    data_cuaca = data_cuaca[['tahun', 'tekanan_udara', 'penyinaran_matahari', 'suhu_minimum', 'suhu_rata_rata', 'suhu_maksimum', 'kecepatan_angin', 'kelembaban', 'hari_hujan', 'curah_hujan']]

    # Drop Duplicated
    col_duplicated = ['tekanan_udara','penyinaran_matahari','suhu_minimum','suhu_rata_rata','suhu_maksimum','kecepatan_angin','kelembaban','hari_hujan','curah_hujan']
    data_drop_duplicated =  data_cuaca.drop_duplicates(subset=col_duplicated,keep=False)
    data_drop_null = data_drop_duplicated.dropna(subset=['curah_hujan'])

    # Drop Missing Value
    missing_columns = data_drop_null.drop(["tahun","curah_hujan"], axis=1).columns
    clean = data_drop_null.loc[:,["tahun","curah_hujan"]]

    # Random Missing Value
    def random_imputation(df, feature):
        number_missing = df[feature].isnull().sum()
        observed_values = df.loc[df[feature].notnull(), feature]
        df.loc[df[feature].isnull(), feature + '_imp'] = np.random.choice(observed_values, number_missing, replace = True)
        
        return df

    for feature in missing_columns:
        data_drop_null[feature + '_imp'] = data_drop_null[feature]
        data_drop_null = random_imputation(data_drop_null, feature)

    deter_data = pd.DataFrame(columns = [name for name in missing_columns])

    for feature in missing_columns:
            
        deter_data[feature] = data_drop_null[feature + "_imp"]
        parameters = list(set(data_drop_null.columns) - set(missing_columns) - {feature + '_imp'})
        
        model = LinearRegression()
        model.fit(X = data_drop_null[parameters], y = data_drop_null[feature + '_imp'])
        
        deter_data.loc[data_drop_null[feature].isnull(), feature] = model.predict(data_drop_null[parameters])[data_drop_null[feature].isnull()]

    data_drop_null = pd.concat([clean, deter_data], axis=1)

    # Outlier Remv
    def find_outlier(df, variable_name):    
        q1 = df[variable_name].quantile(0.25)
        q3 = df[variable_name].quantile(0.75)
        iqr = q3-q1
        outer_fence = 3*iqr
        outer_fence_le = q1-outer_fence
        outer_fence_ue = q3+outer_fence
        return outer_fence_le, outer_fence_ue
    data_outlier = data_drop_null.copy()
    col_outlier = data_outlier.drop(['tahun', 'curah_hujan'], axis=1).columns
    for col in col_outlier:
        outer_fence_le, outer_fence_ue = find_outlier(data_outlier, col)
        data_outlier[col] = np.where(data_outlier[col] < outer_fence_le, outer_fence_le,
                                    np.where(data_outlier[col] > outer_fence_ue, outer_fence_ue, data_outlier[col]))

    # Feature Selection
    data_selection = data_outlier.copy()
    data_selection = data_selection[["penyinaran_matahari", "suhu_rata_rata", "kelembaban", "hari_hujan", "curah_hujan"]]


    # Variabel Independen and Dependen
    X = data_selection.drop("curah_hujan", axis=1)
    y = data_selection["curah_hujan"]
    col_names = X.columns

    # Normalization
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, columns = col_names)
    

    lin_reg = LinearRegression().fit(X, y)

    data = pd.DataFrame(data)
    data = scaler.fit_transform(data)
    return lin_reg.predict(data)

