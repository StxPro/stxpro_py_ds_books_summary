from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
from IPython.core.display import display, HTML
from pandas import DataFrame as DataFrame_pd
from pyspark.sql.dataframe import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

class DataTypes(BaseEstimator, TransformerMixin):

    def __init__(self, target='target'):
        self.target = target
        self.data = None

    def str_if_not_null(self, x):
        if pd.isnull(x) or (x is None) or pd.isna(x) or (x is not x):
            return x
        return str(x)

    def fit(self, data, y=None):
        # data = data.copy()
        try:
            data.drop(self.target, axis=1, inplace=True)
        except:
            pass
        data.columns = [str(i) for i in data.columns]
        data.columns = data.columns.str.replace('[,]','')
        data.columns = data.columns.str.replace(r"[\,\}\{\]\[\:\"\']", "")
        data.replace([np.inf, -np.inf], np.NaN, inplace=True)

        for i in data.select_dtypes(include=["object"]).columns:
            try:            
                data[i] = data[i].astype("int64")
                print('object -> int64 ',i)
            except:
                None

        for i in data.select_dtypes(include=["object"]).columns:
            try:
                print(i)
                data[i] = pd.to_datetime(
                    data[i], infer_datetime_format=True, utc=False, errors="raise"
                )
                print('object -> datetime ',i)
            except:
                continue

        for i in data.select_dtypes(include=["bool", "category"]).columns:
            print('bool, category -> object ',i)
            data[i] = data[i].astype("object")

        for i in data.select_dtypes(include=["float64"]).columns:
            print('float64 -> float32 ',i)
            data[i] = data[i].astype("float32")
            # count how many Nas are there
            na_count = sum(data[i].isnull())
            # count how many digits are there that have decimiles
            count_float = np.nansum(
                [False if r.is_integer() else True for r in data[i]]
            )
            # total decimiels digits
            count_float = (
                count_float - na_count
            )  # reducing it because we know NaN is counted as a float digit
            #print('count_float: ', count_float)
            #print('data[i].nunique(): ', data[i].nunique())
            # now if there isnt any float digit , & unique levales are less than 20 and there are Na's then convert it to object
            if (count_float == 0) & (data[i].nunique() <= 20) & (na_count > 0):
                print('float64 -> float32 -> object (nunique <= 20 & na_count > 0 & count_float == 0)',i)
                data[i] = data[i].astype("object")
        
        for i in data.select_dtypes(include=["int64"]).columns:
            if data[i].nunique() <= 20:  # hard coded
                print('int64 -> object (nunique <= 20)',i)
                data[i] = data[i].apply(self.str_if_not_null)
            else:
                print('int64 -> float32 ',i)
                data[i] = data[i].astype("float32")

        for i in data.select_dtypes(include=["float32"]).columns:
            if data[i].nunique() == 2:
                print('float32 -> object (nunique = 2)',i)
                data[i] = data[i].astype("float64").apply(self.str_if_not_null)

        self.data = data
        return self

    def transform(self, data, y=None):
        data.columns = [str(i) for i in data.columns]
        data.columns = data.columns.str.replace('[,]','')
        data.columns = data.columns.str.replace(r"[\,\}\{\]\[\:\"\']", "")
        data.replace([np.inf, -np.inf], np.NaN, inplace=True)
        dt = self.data.dtypes.copy()
        try:
            data.drop(self.target, axis=1, inplace=True)
        except:
            pass
        return data.astype(dt)

    def fit_transform(self, data, y=None):
        self.fit(data)
        return self.transform(data)


class Simple_Imputer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.statistics = []
        self.numeric_imputer = None
        self.categorical_imputer = None
        self.numeric_columns = None
        self.categorical_columns = None

    def fit(self, data, y=None):
        self.numeric_columns = data.select_dtypes(include=["float32", "int64"]).columns
        self.categorical_columns = data.select_dtypes(include=["object"]).columns

        self.numeric_imputer = SimpleImputer(
            strategy='mean',
            fill_value=None,
        )

        self.categorical_imputer = SimpleImputer(
            strategy='constant',
            fill_value=None,
        )

        if not self.numeric_columns.empty:
            self.numeric_imputer.fit(data[self.numeric_columns])
            self.statistics.append((self.numeric_imputer.statistics_, self.numeric_columns))
        if not self.categorical_columns.empty:
            self.categorical_imputer.fit(data[self.categorical_columns])
            self.statistics.append((self.categorical_imputer.statistics_, self.categorical_columns))

        return self

    def transform(self, data, y=None):
        imputed_data = []
        dt = data.dtypes.copy()
        if not self.numeric_columns.empty:
            numeric_data = pd.DataFrame(
                self.numeric_imputer.transform(data[self.numeric_columns]),
                columns=self.numeric_columns,
                index=data.index,
            )
            imputed_data.append(numeric_data)
        if not self.categorical_columns.empty:
            categorical_data = pd.DataFrame(
                self.categorical_imputer.transform(data[self.categorical_columns]),
                columns=self.categorical_columns,
                index=data.index,
            )
            for col in categorical_data.columns:
                categorical_data[col] = categorical_data[col].apply(str)
            imputed_data.append(categorical_data)

        if imputed_data:
            data.update(pd.concat(imputed_data, axis=1))
        data = data.astype(dt)
        return data

    def fit_transform(self, data, y=None):
        self.fit(data)
        return self.transform(data)


class Dummyfy(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.ohe = None
        self.data_columns = None

    def fit(self, data, y=None):
        self.ohe = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
        categorical_data = data.select_dtypes(include=("object"))
        self.ohe.fit(categorical_data)
        self.data_columns = self.ohe.get_feature_names(categorical_data.columns)
        
        return self

    def transform(self, data, y=None):
        data_nonc = data.select_dtypes(exclude=("object"))
        array = self.ohe.transform(data.select_dtypes(include=("object"))).toarray()
        data_dummies = pd.DataFrame(array, columns=self.data_columns)
        data_dummies.index = data_nonc.index
        return pd.concat((data_nonc, data_dummies), axis=1)

    def fit_transform(self, data, y=None):
        self.fit(data)
        return self.transform(data)


class DropPerfectCorrCols(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.drop_columns = None

    def fit(self, data, y=None):
        corr = pd.DataFrame(np.corrcoef(data.T))
        corr.columns = data.columns
        corr.index = data.columns
        corr_matrix = abs(corr)
        corr_matrix["column"] = corr_matrix.index
        corr_matrix.reset_index(drop=True, inplace=True)
        cols = corr_matrix.column
        melt = corr_matrix.melt(id_vars=["column"], value_vars=cols).sort_values(
            by="value", ascending=False
        )  # .dropna()
        melt["value"] = round(melt["value"], 2)  # round it to two digits
        c1 = melt["value"] == 1.00
        c2 = melt["column"] != melt["variable"]
        melt = melt[((c1 == True) & (c2 == True))]
        melt["all_columns"] = melt["column"] + melt["variable"]
        melt["all_columns"] = [sorted(i) for i in melt["all_columns"]]
        melt = melt.sort_values(by="all_columns")
        melt = melt.iloc[::2, :]
        self.drop_columns = melt["variable"]
        
        return self

    def transform(self, data, y=None):
        return data.drop(self.drop_columns, axis=1)

    def fit_transform(self, data, y=None):
        self.fit(data)
        return self.transform(data)




def __fix_types_pprint(x):
    try:
        x.str
        return x
    except:
        try:
            if 'int' in str(x.dtype):
                return x.apply(lambda x: f'{x:,}')
            return x.astype(np.float64)
        except:
            return x


def __insert_style(background=None, color=None):
    """
    Returns css style
    :param background: string
    :param color: string
    :return: HTML style
    """
    background = background or 'white'
    color = color or 'black'
    style = """
        <style scoped>
            .dataframe-div {
                overflow: auto;
                position: relative;
            }

            .dataframe-div .dataframe thead th {
                position: -webkit-sticky; /* for Safari */
                position: sticky;
                top: 0;
                background: """+background+""";
                color: """+color+""";
            }

            .dataframe-div .dataframe thead th:first-child {
                left: 0;
                z-index: 1;
            }

            .dataframe-div .dataframe tbody tr th:only-of-type {
                    vertical-align: middle;
                }

            .dataframe-div .dataframe tbody tr th {
                position: sticky;
                left: 0;
                background: """+background+""";
                color: """+color+""";
                vertical-align: top;
            }

            }
        </style>
        """
    return HTML(style)

def fix_to_pandas(limit=5, background='#00734d', color='white', max_columns=500):
    """
    Lets you improve the toPandas style of a DataFrame,
    allows modifying the method DataFrame.toPandas()
    :param color: headers color with pprint function
    :param background: background headers color with pprint function
    :param limit: number of records to display in a Pandas DataFrame
    :param max_columns: number of columns to display in a Pandas DataFrame
    """
    pd.set_option('display.max_columns', max_columns)
    pd.set_option('display.width', 1000)
    pd.options.display.float_format = '{:20,.3f}'.format

    def toPandas(self, n=limit):
        return self.limit(n).toPandas()
    DataFrame.toPandas_ = toPandas

    def pprint(self, limit=5, index=None, size=300):
        size = str(size)
        if index is None:
            df_html = self.toPandas_(limit).apply(__fix_types_pprint, axis=0).to_html()
        else:
            df_html = (self.toPandas_(limit).set_index(index)
                       .rename_axis(index=None, columns=None).rename_axis(index, axis=1)
                       .apply(__fix_types_pprint, axis=0).to_html())
        df_html = '<div class="dataframe-div" style="max-height: {0}px;">{1}\n</div>'.format(size, df_html)
        return display(HTML(df_html))
    DataFrame.pprint = pprint

    def pprint2(self, index=None, size=800):
        size = str(size)
        if index is None:
            df_html = self.toPandas().apply(__fix_types_pprint, axis=0).to_html()
        else:
            df_html = (self.toPandas().set_index(index)
                       .rename_axis(index=None, columns=None).rename_axis(index, axis=1)
                       .apply(__fix_types_pprint, axis=0).to_html())
        df_html = '<div class="dataframe-div" style="max-height: {0}px;">{1}\n</div>'.format(size, df_html)
        return display(HTML(df_html))
    DataFrame.pprint2 = pprint2

    def pprint_pd(self, limit=5, index=None, size=300):
        size = str(size)
        if index is None:
            df_html = self.head(limit).apply(__fix_types_pprint, axis=0).to_html()
        else:
            df_html = (self.head(limit).set_index(index)
                        .rename_axis(index=None, columns=None).rename_axis(index, axis=1)
                        .apply(__fix_types_pprint, axis=0).to_html())
        df_html = '<div class="dataframe-div" style="max-height: {0}px;">{1}\n</div>'.format(size, df_html)
        return display(HTML(df_html))
    DataFrame_pd.pprint = pprint_pd        

    def pprint_pd2(self, index=None, size=800):
        size = str(size)
        if index is None:
            df_html = self.apply(__fix_types_pprint, axis=0).to_html()
        else:
            df_html = (self.set_index(index)
                        .rename_axis(index=None, columns=None).rename_axis(index, axis=1)
                        .apply(__fix_types_pprint, axis=0).to_html())
        df_html = '<div class="dataframe-div" style="max-height: {0}px;">{1}\n</div>'.format(size, df_html)
        return display(HTML(df_html))
    DataFrame_pd.pprint2 = pprint_pd2        

    # Ejecutar este import ocasiona problemas con el método enable_logging
    # Se recomienda ejecutar primero el enable_logging antes de este método
    try:
        from pyspark.pandas import DataFrame as DataFrame_ps
        DataFrame_ps.pprint = pprint_pd
        DataFrame_ps.pprint2 = pprint_pd2
    except:
        pass

    display(__insert_style(background, color))

__show_method = None

def fix_show(n=5, truncate=False):
    """
    Lets you improve the show style of a DataFrame,
    allows modifying the default parameters of the DataFrame.show()
    :param n: represents the n parameter of the DataFrame.show(n=n)
    :param truncate: represents the truncate parameter of the DataFrame.show(truncate=truncate)
    """
    global __show_method
    if __show_method is None:
        DataFrame.show_old = DataFrame.show
        __show_method = DataFrame.show_old
    else:
        DataFrame.show_old = __show_method

    def show(self, n=n, truncate=truncate):
        return self.show_old(n, truncate)
    DataFrame.show = show
    style = """<style>
            .output_subarea.output_text.output_stream.output_stdout > pre {
              width:max-content;
            }

            .p-Widget.jp-RenderedText.jp-OutputArea-output > pre {
              width:max-content;
            }
            </style>"""
    display(HTML(style))
