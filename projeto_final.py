# Importações
import streamlit as st
import pandas as pd
from pycaret.classification import setup, load_model, predict_model
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle
import io
import os

# Caminho absoluto do modelo
caminho_absoluto_modelo = os.path.join(os.getcwd(), "melhor_modelo.pkl")

# Função principal do Streamlit
def main():
    # Configuração inicial da página
    st.set_page_config(page_title="Escoragem com PyCaret", layout="wide")
    st.title("Escoragem de Modelos com PyCaret")

    # Verificar se o arquivo do modelo existe
    if not os.path.isfile(caminho_absoluto_modelo):
        st.error(f"Erro: Arquivo do modelo '{caminho_absoluto_modelo}' não encontrado no diretório atual.")
        return

    # Carregando o modelo treinado
    st.write("**Carregando o Modelo Treinado...**")
    try:
        modelo = load_model(caminho_absoluto_modelo.replace(".pkl", ""))
        st.success("Modelo carregado com sucesso!")
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {str(e)}")
        return

    # Upload de arquivo CSV
    st.sidebar.header("Upload do Dataset")
    arquivo_csv = st.sidebar.file_uploader(
        "Suba seu arquivo CSV para análise", type=["csv"]
    )

    if arquivo_csv:
        try:
            # Lendo o arquivo CSV
            st.write("**Visualizando os Dados Carregados:**")
            df = pd.read_csv(arquivo_csv)

            # Garantindo que 'data_ref' esteja no formato datetime, se existir
            if 'data_ref' in df.columns:
                df['data_ref'] = pd.to_datetime(df['data_ref'], errors='coerce')
                if df['data_ref'].isnull().any():
                    st.warning("Alguns valores na coluna 'data_ref' não puderam ser convertidos para datetime.")
                else:
                    st.success("Coluna 'data_ref' processada como datetime com sucesso.")

            st.write(df.head())

            # Realizando previsões
            st.write("**Realizando Previsões...**")
            previsoes = predict_model(modelo, data=df)

            # Exibindo resultados
            st.write("**Resultados das Previsões:**")
            st.write(previsoes.head())

            # Botão para baixar os resultados
            st.download_button(
                label="📥 Baixar Resultados",
                data=previsoes.to_csv(index=False).encode("utf-8"),
                file_name="previsoes.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {str(e)}")
    else:
        st.info("Por favor, faça o upload de um arquivo CSV para continuar.")


# Execução principal
if __name__ == "__main__":
    main()
