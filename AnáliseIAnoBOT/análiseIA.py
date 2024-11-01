import streamlit as st
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import re

st.title("Análise Comparativa de Interações com Cliente - 2023 vs. 2024")
st.write("Esta aplicação compara métricas de interações com clientes entre os períodos de 2023 e 2024.")

# Upload dos arquivos
outubro2023 = st.file_uploader("Upload do arquivo de outubro de 2023", type="csv")
outubro2024 = st.file_uploader("Upload do arquivo de outubro de 2024", type="csv")

# Funções de Análise
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    if pd.isna(text):
        return None
    blob = TextBlob(text)
    textblob_polarity = blob.sentiment.polarity
    vader_scores = analyzer.polarity_scores(text)
    return (textblob_polarity + vader_scores['compound']) / 2

def extract_intentions(transcript_column):
    intentions = []
    for text in transcript_column.dropna():
        found_intentions = re.findall(r'\b(intenção\s+\w+|intenção:\s*\w+)', text, re.IGNORECASE)
        intentions.extend(found_intentions)
    return intentions

def categorize_duration_adjusted(duration):
    if duration < 30:
        return 'curto'
    elif duration <= 120:
        return 'medio'
    else:
        return 'longo'

# Análises
if outubro2023 and outubro2024:
    data_2023 = pd.read_csv(outubro2023)
    data_2024 = pd.read_csv(outubro2024)
    
    # CSAT
    data_2023['surveyAnswerPostSurvey'] = pd.to_numeric(data_2023['surveyAnswerPostSurvey'], errors='coerce')
    data_2024['surveyAnswerPostSurvey'] = pd.to_numeric(data_2024['surveyAnswerPostSurvey'], errors='coerce')
    data_filtered_csat_2023 = data_2023.dropna(subset=['surveyAnswerPostSurvey'])
    data_filtered_csat_2024 = data_2024.dropna(subset=['surveyAnswerPostSurvey'])
    csat_mean_2023 = data_filtered_csat_2023['surveyAnswerPostSurvey'].mean()
    csat_mean_2024 = data_filtered_csat_2024['surveyAnswerPostSurvey'].mean()
    
    # Sentimento
    data_filtered_csat_2023['sentiment_score'] = data_filtered_csat_2023['transcriptAll'].apply(analyze_sentiment)
    data_filtered_csat_2024['sentiment_score'] = data_filtered_csat_2024['transcriptAll'].apply(analyze_sentiment)
    sentiment_mean_2023 = data_filtered_csat_2023['sentiment_score'].mean()
    sentiment_mean_2024 = data_filtered_csat_2024['sentiment_score'].mean()
    
    # Tempo Médio de Primeira Resposta
    data_2023['firstResponseTimeAgentFromConsumer'] = pd.to_numeric(data_2023['firstResponseTimeAgentFromConsumer'], errors='coerce')
    data_2024['firstResponseTimeAgentFromConsumer'] = pd.to_numeric(data_2024['firstResponseTimeAgentFromConsumer'], errors='coerce')
    first_response_mean_2023 = data_2023['firstResponseTimeAgentFromConsumer'].mean()
    first_response_mean_2024 = data_2024['firstResponseTimeAgentFromConsumer'].mean()
    
    # Taxa de Escalamento para Humanos
    data_2023['messageCountAgentBot'] = data_2023['messageCountAgentBot'].fillna(0)
    data_2023['messageCountAgentHuman'] = data_2023['messageCountAgentHuman'].fillna(0)
    data_2024['messageCountAgentBot'] = data_2024['messageCountAgentBot'].fillna(0)
    data_2024['messageCountAgentHuman'] = data_2024['messageCountAgentHuman'].fillna(0)
    resolved_by_ia_2023 = data_2023[data_2023['messageCountAgentBot'] > 0].shape[0]
    escalated_to_human_2023 = data_2023[data_2023['messageCountAgentHuman'] > 0].shape[0]
    resolved_by_ia_2024 = data_2024[data_2024['messageCountAgentBot'] > 0].shape[0]
    escalated_to_human_2024 = data_2024[data_2024['messageCountAgentHuman'] > 0].shape[0]
    
    # Frequência de Intenções
    intentions_2023 = extract_intentions(data_2023['transcriptAll'])
    intentions_2024 = extract_intentions(data_2024['transcriptAll'])
    intentions_count_2023 = Counter(intentions_2023).most_common(10)
    intentions_count_2024 = Counter(intentions_2024).most_common(10)
    
    # Eficiência do Canal
    channel_counts_2023 = data_2023['source'].value_counts()
    channel_counts_2024 = data_2024['source'].value_counts()
    
    # Número Médio de Mensagens
    data_2023['messageCountAgent'] = data_2023['messageCountAgent'].fillna(0)
    data_2023['messageCountConsumer'] = data_2023['messageCountConsumer'].fillna(0)
    data_2024['messageCountAgent'] = data_2024['messageCountAgent'].fillna(0)
    data_2024['messageCountConsumer'] = data_2024['messageCountConsumer'].fillna(0)
    data_2023['totalMessageCount'] = data_2023['messageCountAgent'] + data_2023['messageCountConsumer']
    data_2024['totalMessageCount'] = data_2024['messageCountAgent'] + data_2024['messageCountConsumer']
    average_messages_2023 = data_2023['totalMessageCount'].mean()
    average_messages_2024 = data_2024['totalMessageCount'].mean()
    
    # Taxa de Resolução na Primeira Interação (FCR)
    resolved_first_attempt_2023 = data_2023[data_2023['closeReason'] == 'AGENT'].shape[0]
    fcr_rate_2023 = (resolved_first_attempt_2023 / data_2023.shape[0]) * 100
    resolved_first_attempt_2024 = data_2024[data_2024['closeReason'] == 'AGENT'].shape[0]
    fcr_rate_2024 = (resolved_first_attempt_2024 / data_2024.shape[0]) * 100
    
    # Distribuição de Tempos de Interação
    data_2023['startTimeUTC'] = pd.to_datetime(data_2023['startTimeUTC'], errors='coerce')
    data_2023['endTimeUTC'] = pd.to_datetime(data_2023['endTimeUTC'], errors='coerce')
    data_2024['startTimeUTC'] = pd.to_datetime(data_2024['startTimeUTC'], errors='coerce')
    data_2024['endTimeUTC'] = pd.to_datetime(data_2024['endTimeUTC'], errors='coerce')
    data_2023['duration_minutes'] = (data_2023['endTimeUTC'] - data_2023['startTimeUTC']).dt.total_seconds() / 60
    data_2024['duration_minutes'] = (data_2024['endTimeUTC'] - data_2024['startTimeUTC']).dt.total_seconds() / 60
    data_filtered = data_2023.dropna(subset=['duration_minutes']).copy()
    data_2024_filtered = data_2024.dropna(subset=['duration_minutes']).copy()
    data_filtered['duration_category'] = data_filtered['duration_minutes'].apply(categorize_duration_adjusted)
    data_2024_filtered['duration_category'] = data_2024_filtered['duration_minutes'].apply(categorize_duration_adjusted)
    duration_distribution_adjusted_2023 = data_filtered['duration_category'].value_counts(normalize=True) * 100
    duration_distribution_adjusted_2024 = data_2024_filtered['duration_category'].value_counts(normalize=True) * 100

    # Exibindo os Resultados Comparativos
    st.subheader("Comparação dos Resultados entre Outubro 2023(sem IA) e 2024(Com IA)")
    
    st.write("### Média de Satisfação")
    st.write(f"2023: {csat_mean_2023:.2f} | 2024: {csat_mean_2024:.2f}")
    
    st.write("### Sentimento Médio")
    st.write(f"2023: {sentiment_mean_2023:.2f} | 2024: {sentiment_mean_2024:.2f}")
    
    st.write("### Tempo Médio de Primeira Resposta (min)")
    st.write(f"2023: {first_response_mean_2023:.2f} | 2024: {first_response_mean_2024:.2f}")
    
    st.write("### Interações Resolvidas pela IA e Escaladas para Humanos")
    st.write(f"Resolvidas pela IA 2023: {resolved_by_ia_2023} | 2024: {resolved_by_ia_2024}")
    st.write(f"Escaladas para Humanos 2023: {escalated_to_human_2023} | 2024: {escalated_to_human_2024}")
    
    st.write("### Número Médio de Mensagens por Interação")
    st.write(f"2023: {average_messages_2023:.2f} | 2024: {average_messages_2024:.2f}")
    
    st.write("### Taxa de Resolução na Primeira Interação (FCR)")
    st.write(f"2023: {fcr_rate_2023:.2f}% | 2024: {fcr_rate_2024:.2f}%")
    
    st.write("### Distribuição de Tempos de Interação (em %)")
    st.write("2023:")
    st.write(duration_distribution_adjusted_2023)
    st.write("2024:")
    st.write(duration_distribution_adjusted_2024)
    
    st.write("### Entrada de mensgagens")
    st.write("2023:")
    st.write(channel_counts_2023)
    st.write("2024:")
    st.write(channel_counts_2024)

else:
    st.write("Por favor, faça o upload dos arquivos de dados para prosseguir com a análise.")
