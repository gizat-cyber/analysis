import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(
    page_title="Анализ данных о найме сотрудников",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок приложения
st.title("👥 Анализ данных о найме сотрудников")
st.markdown("---")

# Функция для автоматической загрузки встроенных данных
@st.cache_data
def load_builtin_data():
    """Загружает встроенный CSV файл с данными о найме"""
    try:
        # Пытаемся загрузить встроенный файл
        csv_file = "merge-csv.com__68b9ee302f5dd.csv"
        
        # Пробуем разные кодировки
        encodings = ['utf-8', 'latin1', 'cp1251']
        df = None
        
        for encoding in encodings:
            try:
                # Пропускаем первые 3 строки (комментарии) и используем 4-ю как заголовки
                df = pd.read_csv(
                    csv_file, 
                    encoding=encoding,
                    skiprows=3,
                    header=0,
                    engine='python'
                )
                st.success(f"✅ Встроенные данные загружены с кодировкой: {encoding}")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                st.warning(f"Попытка с кодировкой {encoding} не удалась: {e}")
                continue
        
        if df is not None:
            return df
        else:
            st.error("❌ Не удалось загрузить встроенные данные")
            return None
            
    except Exception as e:
        st.error(f"❌ Ошибка при загрузке встроенных данных: {e}")
        return None

# Функция для загрузки и обработки данных
@st.cache_data
def load_data(uploaded_file):
    """Загружает CSV файл и возвращает DataFrame"""
    try:
        if uploaded_file is not None:
            # Пробуем разные кодировки
            encodings = ['utf-8', 'latin1', 'cp1251']
            for encoding in encodings:
                try:
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    return df
                except UnicodeDecodeError:
                    continue
            st.error("Не удалось прочитать файл. Попробуйте другой формат кодировки.")
            return None
    except Exception as e:
        st.error(f"Ошибка при загрузке файла: {e}")
        return None

# Функция для анализа данных
def analyze_data(df):
    """Проводит базовый анализ данных"""
    st.subheader("📊 Общая информация о данных")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Общее количество записей", len(df))
    
    with col2:
        st.metric("Количество столбцов", len(df.columns))
    
    with col3:
        missing_values = df.isnull().sum().sum()
        st.metric("Пропущенные значения", missing_values)
    
    with col4:
        memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Размер данных (МБ)", f"{memory_usage:.2f}")
    
    # Информация о столбцах
    st.subheader("📋 Структура данных")
    
    # Создаем DataFrame с правильными типами данных
    col_info_data = []
    for col in df.columns:
        try:
            missing_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            dtype_str = str(df[col].dtype)
            
            # Берем только первые 3 примера для избежания проблем с Arrow
            sample_values = df[col].dropna().head(3).astype(str).tolist()
            sample_str = ", ".join(sample_values) if sample_values else "N/A"
            
            col_info_data.append({
                'Столбец': col,
                'Тип данных': dtype_str,
                'Пропущено': missing_count,
                'Уникальных': unique_count,
                'Примеры': sample_str
            })
        except Exception as e:
            col_info_data.append({
                'Столбец': col,
                'Тип данных': 'Ошибка',
                'Пропущено': 0,
                'Уникальных': 0,
                'Примеры': f'Ошибка: {str(e)}'
            })
    
    col_info = pd.DataFrame(col_info_data)
    st.dataframe(col_info, width='stretch')
    
    return col_info

# Функция для детального анализа найма
def detailed_hiring_analysis(df):
    """Детальный анализ найма с фокусом на источники и успешность"""
    st.subheader("🎯 Детальный анализ найма")
    
    # Поиск столбцов, связанных с наймом
    hiring_columns = []
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['hire', 'найм', 'принят', 'статус', 'результат', 'outcome', 'status']):
            hiring_columns.append(col)
    
    if hiring_columns:
        st.write(f"✅ Найдены столбцы найма: {hiring_columns}")
        
        # Анализ основного столбца найма
        main_hiring_col = hiring_columns[0]
        st.write(f"**Анализируем столбец:** {main_hiring_col}")
        
        # Распределение статусов найма
        hiring_dist = df[main_hiring_col].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Распределение статусов найма:**")
            st.dataframe(hiring_dist)
        
        with col2:
            fig = px.pie(
                values=hiring_dist.values,
                names=hiring_dist.index,
                title="Распределение статусов найма"
            )
            st.plotly_chart(fig, width="stretch")
        
        # Анализ успешных кандидатов (Active/Approved)
        st.subheader("🏆 Анализ успешных кандидатов")
        
        # Определяем успешные статусы
        success_keywords = ['active', 'approved', 'найм', 'принят', 'успех']
        success_statuses = []
        
        for status in hiring_dist.index:
            status_lower = str(status).lower()
            if any(keyword in status_lower for keyword in success_keywords):
                success_statuses.append(status)
        
        if success_statuses:
            st.write(f"**Успешные статусы:** {success_statuses}")
            
            # Фильтруем успешных кандидатов
            successful_df = df[df[main_hiring_col].isin(success_statuses)]
            st.write(f"**Количество успешных кандидатов:** {len(successful_df)}")
            
            # Анализ по должностям для успешных
            if 'Worklist' in df.columns:
                st.write("**Должности успешных кандидатов:**")
                worklist_success = successful_df['Worklist'].value_counts()
                
                fig = px.bar(
                    x=worklist_success.values,
                    y=worklist_success.index,
                    title="Должности успешных кандидатов",
                    orientation='h'
                )
                st.plotly_chart(fig, width="stretch")
            
            # Анализ по штатам для успешных
            if 'State' in df.columns:
                st.write("**География успешных кандидатов:**")
                state_success = successful_df['State'].value_counts().head(10)
                
                fig = px.bar(
                    x=state_success.values,
                    y=state_success.index,
                    title="Топ-10 штатов успешных кандидатов",
                    orientation='h'
                )
                st.plotly_chart(fig, width="stretch")
        
        # Анализ по источникам найма
        st.subheader("📍 Анализ источников найма")
        
        # Поиск столбцов с источниками
        source_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['source', 'источник', 'recruiter', 'рекрутер']):
                source_columns.append(col)
        
        if source_columns:
            st.write(f"**Столбцы источников:** {source_columns}")
            
            for source_col in source_columns:
                st.write(f"**Анализ столбца:** {source_col}")
                
                # Распределение по источникам
                source_dist = df[source_col].value_counts().head(10)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Топ-10 источников:**")
                    st.dataframe(source_dist)
                
                with col2:
                    fig = px.pie(
                        values=source_dist.values,
                        names=source_dist.index,
                        title=f"Распределение по {source_col}"
                    )
                    st.plotly_chart(fig, width="stretch")
                
                # Эффективность источников (отношение успешных к общему количеству)
                if success_statuses:
                    st.write("**Эффективность источников (отношение успешных):**")
                    
                    source_effectiveness = {}
                    for source in source_dist.index:
                        if pd.notna(source) and source != "":
                            total_from_source = len(df[df[source_col] == source])
                            successful_from_source = len(df[(df[source_col] == source) & (df[main_hiring_col].isin(success_statuses))])
                            effectiveness = (successful_from_source / total_from_source) * 100 if total_from_source > 0 else 0
                            source_effectiveness[source] = effectiveness
                    
                    # Сортируем по эффективности
                    sorted_effectiveness = dict(sorted(source_effectiveness.items(), key=lambda x: x[1], reverse=True))
                    
                    fig = px.bar(
                        x=list(sorted_effectiveness.values()),
                        y=list(sorted_effectiveness.keys()),
                        title="Эффективность источников найма (%)",
                        orientation='h'
                    )
                    st.plotly_chart(fig, width="stretch")
        
        # Анализ по времени
        st.subheader("⏰ Временной анализ найма")
        
        time_columns = [col for col in df.columns if any(keyword in col.lower() 
                       for keyword in ['date', 'дата', 'время', 'time', 'год', 'year'])]
        
        if time_columns:
            st.write(f"**Временные столбцы:** {time_columns}")
            
            for time_col in time_columns:
                try:
                    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                    df_time = df.dropna(subset=[time_col])
                    
                    if len(df_time) > 0:
                        st.write(f"**Анализ столбца:** {time_col}")
                        
                        # Группировка по месяцам
                        df_time['Месяц'] = df_time[time_col].dt.to_period('M')
                        monthly_data = df_time.groupby(['Месяц', main_hiring_col]).size().unstack(fill_value=0)
                        
                        # Показываем только последние 24 месяца
                        recent_months = monthly_data.tail(24)
                        
                        fig = px.line(
                            recent_months,
                            title=f"Тренд найма по месяцам ({time_col})",
                            labels={'value': 'Количество', 'index': 'Месяц'}
                        )
                        st.plotly_chart(fig, width="stretch")
                        
                        # Анализ по годам
                        df_time['Год'] = df_time[time_col].dt.year
                        yearly_data = df_time.groupby(['Год', main_hiring_col]).size().unstack(fill_value=0)
                        
                        fig = px.bar(
                            yearly_data,
                            title=f"Распределение по годам ({time_col})",
                            barmode='group'
                        )
                        st.plotly_chart(fig, width="stretch")
                        
                except Exception as e:
                    st.write(f"Не удалось проанализировать {time_col}: {e}")
    
    else:
        st.warning("Не найдены столбцы найма. Показываем все столбцы:")
        selected_col = st.selectbox("Выберите столбец для анализа:", df.columns)
        
        if selected_col:
            col_dist = df[selected_col].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Распределение значений:")
                st.dataframe(col_dist)
            
            with col2:
                fig = px.pie(
                    values=col_dist.values,
                    names=col_dist.index,
                    title=f"Распределение в столбце {selected_col}"
                )
                st.plotly_chart(fig, width="stretch")

# Функция для анализа продолжительности работы
def analyze_tenure(df):
    """Анализирует продолжительность работы сотрудников"""
    st.subheader("⏱️ Анализ продолжительности работы")
    
    # Поиск столбцов с продолжительностью
    tenure_columns = []
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['tenure', 'стаж', 'длительность', 'duration', 'месяц', 'месяцев', 'лет']):
            tenure_columns.append(col)
    
    if tenure_columns:
        st.write(f"Найдены столбцы с продолжительностью работы: {tenure_columns}")
        
        for col in tenure_columns:
            st.write(f"**Анализ столбца: {col}**")
            
            # Статистики
            tenure_stats = df[col].describe()
            st.write("Статистики продолжительности работы:")
            st.dataframe(tenure_stats)
            
            # Гистограмма
            fig = px.histogram(
                df, 
                x=col, 
                title=f"Распределение продолжительности работы ({col})",
                labels={'x': col, 'y': 'Количество'}
            )
            st.plotly_chart(fig, width="stretch")
            
            # Анализ по группам (если есть столбец найма)
            hiring_columns = [c for c in df.columns if any(keyword in c.lower() 
                           for keyword in ['hire', 'найм', 'принят', 'статус'])]
            
            if hiring_columns:
                hiring_col = hiring_columns[0]
                fig = px.box(
                    df, 
                    x=hiring_col, 
                    y=col,
                    title=f"Продолжительность работы по результатам найма"
                )
                st.plotly_chart(fig, width="stretch")
    else:
        st.warning("Не найдены столбцы с продолжительностью работы")

# Функция для построения модели машинного обучения
def build_ml_model(df):
    """Строит модель машинного обучения для предсказания найма"""
    st.subheader("🤖 Машинное обучение: Предсказание найма")
    
    # Выбор целевой переменной
    st.write("Выберите столбец для предсказания (целевая переменная):")
    target_col = st.selectbox("Целевая переменная:", df.columns)
    
    if target_col:
        # Подготовка данных
        st.write(f"Подготовка данных для предсказания: {target_col}")
        
        # Удаление строк с пропущенными значениями в целевой переменной
        df_clean = df.dropna(subset=[target_col])
        
        # Проверяем количество уникальных значений в целевой переменной
        unique_targets = df_clean[target_col].nunique()
        st.write(f"**Уникальных значений в целевой переменной:** {unique_targets}")
        
        if unique_targets < 2:
            st.error("❌ Недостаточно уникальных значений для построения модели (нужно минимум 2)")
            return
        
        # Проверяем минимальное количество записей для каждого класса
        target_counts = df_clean[target_col].value_counts()
        min_class_size = target_counts.min()
        
        if min_class_size < 2:
            st.error(f"❌ Некоторые классы имеют менее 2 записей. Минимальный размер класса: {min_class_size}")
            st.write("**Распределение классов:**")
            st.dataframe(target_counts, width='stretch')
            return
        
        st.write(f"**Минимальный размер класса:** {min_class_size}")
        
        # Кодирование категориальных переменных
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        # Создание копии для кодирования
        df_encoded = df_clean.copy()
        
        label_encoders = {}
        for col in categorical_cols:
            if col != target_col:
                try:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    label_encoders[col] = le
                except Exception as e:
                    st.warning(f"Не удалось закодировать столбец {col}: {e}")
                    # Удаляем проблемный столбец
                    df_encoded = df_encoded.drop(columns=[col])
        
        # Кодирование целевой переменной
        target_encoder = LabelEncoder()
        df_encoded[target_col] = target_encoder.fit_transform(df_encoded[target_col].astype(str))
        
        # Выбор признаков
        feature_cols = [col for col in df_encoded.columns if col != target_col]
        
        if len(feature_cols) > 0:
            X = df_encoded[feature_cols]
            y = df_encoded[target_col]
            
            st.write(f"**Количество признаков:** {len(feature_cols)}")
            st.write(f"**Размер данных:** {len(X)} записей")
            
            # Проверяем, можно ли использовать stratify
            can_stratify = all(target_counts >= 2)
            
            if can_stratify:
                # Разделение на обучающую и тестовую выборки
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            else:
                # Разделение без stratify
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                st.warning("⚠️ Используется разделение без stratify из-за недостаточного количества данных в некоторых классах")
            
            print(f"📊 Размер обучающей выборки: {len(X_train)}")
            print(f"📊 Размер тестовой выборки: {len(X_test)}")
            
            # Обучение модели
            try:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Предсказания
                y_pred = model.predict(X_test)
                
                # Оценка модели
                st.write("**Результаты модели:**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Отчет о классификации:")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df, width='stretch')
                
                with col2:
                    st.write("Матрица ошибок:")
                    cm = confusion_matrix(y_test, y_pred)
                    fig = px.imshow(
                        cm,
                        labels=dict(x="Предсказанные", y="Фактические"),
                        x=target_encoder.classes_,
                        y=target_encoder.classes_,
                        title="Матрица ошибок"
                    )
                    st.plotly_chart(fig, width='stretch')
                
                # Важность признаков
                feature_importance = pd.DataFrame({
                    'Признак': feature_cols,
                    'Важность': model.feature_importances_
                }).sort_values('Важность', ascending=False)
                
                st.write("**Важность признаков:**")
                fig = px.bar(
                    feature_importance.head(10),
                    x='Важность',
                    y='Признак',
                    title="Топ-10 важных признаков",
                    orientation='h'
                )
                st.plotly_chart(fig, width='stretch')
                
                # Возможность предсказания для новых данных
                st.write("**Предсказание для новых данных:**")
                st.write("Введите значения признаков для предсказания:")
                
                # Создание формы для ввода данных
                input_data = {}
                cols_per_row = 3
                
                for i, col in enumerate(feature_cols[:10]):  # Ограничиваем первыми 10 признаками
                    if i % cols_per_row == 0:
                        cols = st.columns(cols_per_row)
                    
                    with cols[i % cols_per_row]:
                        if col in categorical_cols:
                            unique_vals = df_clean[col].unique()
                            input_data[col] = st.selectbox(f"{col}:", unique_vals)
                        else:
                            input_data[col] = st.number_input(f"{col}:", value=float(df_clean[col].mean()))
                
                if st.button("Сделать предсказание"):
                    # Подготовка входных данных
                    input_df = pd.DataFrame([input_data])
                    
                    # Кодирование категориальных переменных
                    for col in categorical_cols:
                        if col in input_data and col in label_encoders:
                            try:
                                input_df[col] = label_encoders[col].transform([input_data[col]])[0]
                            except:
                                st.error(f"Ошибка кодирования для {col}")
                                continue
                    
                    # Предсказание
                    try:
                        prediction = model.predict(input_df)[0]
                        prediction_proba = model.predict_proba(input_df)[0]
                        
                        # Обратное преобразование
                        predicted_class = target_encoder.inverse_transform([prediction])[0]
                        
                        st.success(f"**Результат предсказания:** {predicted_class}")
                        st.write("**Вероятности классов:**")
                        
                        proba_df = pd.DataFrame({
                            'Класс': target_encoder.classes_,
                            'Вероятность': prediction_proba
                        }).sort_values('Вероятность', ascending=False)
                        
                        st.dataframe(proba_df, width='stretch')
                    except Exception as e:
                        st.error(f"Ошибка при предсказании: {e}")
                
            except Exception as e:
                st.error(f"Ошибка при обучении модели: {e}")
                
        else:
            st.error("Недостаточно признаков для построения модели")
    else:
        st.warning("Выберите целевую переменную для анализа")

# Функция для расширенного анализа данных
def advanced_data_analysis(df):
    """Расширенный анализ данных с дополнительными метриками"""
    st.subheader("🔍 Расширенный анализ данных")
    
    # Анализ качества данных
    st.subheader("📊 Качество данных")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_cells = df.shape[0] * df.shape[1]
        missing_percentage = (df.isnull().sum().sum() / total_cells) * 100
        st.metric("Пропущенные данные", f"{missing_percentage:.1f}%")
    
    with col2:
        duplicate_rows = df.duplicated().sum()
        st.metric("Дублирующиеся строки", duplicate_rows)
    
    with col3:
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        st.metric("Числовые столбцы", numeric_cols)
    
    with col4:
        categorical_cols = len(df.select_dtypes(include=['object']).columns)
        st.metric("Категориальные столбцы", categorical_cols)
    
    # Анализ по штатам
    if 'State' in df.columns:
        st.subheader("🗺️ Географический анализ")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Топ штатов по количеству кандидатов
            state_counts = df['State'].value_counts().head(15)
            fig = px.bar(
                x=state_counts.values,
                y=state_counts.index,
                title="Топ-15 штатов по количеству кандидатов",
                orientation='h'
            )
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            # Карта США (если есть координаты)
            st.write("**Распределение по штатам:**")
            state_summary = pd.DataFrame({
                'Штат': state_counts.index,
                'Кандидаты': state_counts.values,
                'Процент': (state_counts.values / len(df)) * 100
            })
            st.dataframe(state_summary, width="stretch")
    
    # Анализ по времени
    if 'Last App Date' in df.columns:
        st.subheader("⏰ Временной анализ")
        
        try:
            df['Last App Date'] = pd.to_datetime(df['Last App Date'], errors='coerce')
            df_time = df.dropna(subset=['Last App Date'])
            
            if len(df_time) > 0:
                # Анализ по дням недели
                df_time['День недели'] = df_time['Last App Date'].dt.day_name()
                df_time['Месяц'] = df_time['Last App Date'].dt.month_name()
                df_time['Год'] = df_time['Last App Date'].dt.year
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Дни недели
                    day_counts = df_time['День недели'].value_counts()
                    fig = px.pie(
                        values=day_counts.values,
                        names=day_counts.index,
                        title="Активность по дням недели"
                    )
                    st.plotly_chart(fig, width="stretch")
                
                with col2:
                    # Месяцы
                    month_counts = df_time['Месяц'].value_counts()
                    fig = px.bar(
                        x=month_counts.index,
                        y=month_counts.values,
                        title="Активность по месяцам"
                    )
                    st.plotly_chart(fig, width="stretch")
                
                # Тренды по годам
                yearly_trend = df_time['Год'].value_counts().sort_index()
                fig = px.line(
                    x=yearly_trend.index,
                    y=yearly_trend.values,
                    title="Тренд найма по годам",
                    labels={'x': 'Год', 'y': 'Количество заявок'}
                )
                st.plotly_chart(fig, width="stretch")
                
        except Exception as e:
            st.warning(f"Не удалось проанализировать временные данные: {e}")

# Функция для анализа эффективности найма
def hiring_effectiveness_analysis(df):
    """Анализ эффективности найма и факторов успеха"""
    st.subheader("🎯 Анализ эффективности найма")
    
    # Поиск столбца статуса
    status_col = None
    for col in df.columns:
        if 'status' in col.lower():
            status_col = col
            break
    
    if status_col:
        st.write(f"**Анализируем эффективность по столбцу:** {status_col}")
        
        # Определяем успешные статусы
        success_patterns = ['active', 'approved', 'hired', 'success']
        success_statuses = []
        
        for status in df[status_col].unique():
            if pd.notna(status):
                status_lower = str(status).lower()
                if any(pattern in status_lower for pattern in success_patterns):
                    success_statuses.append(status)
        
        if success_statuses:
            st.write(f"**Успешные статусы:** {success_statuses}")
            
            # Общая эффективность
            total_candidates = len(df)
            successful_candidates = len(df[df[status_col].isin(success_statuses)])
            overall_effectiveness = (successful_candidates / total_candidates) * 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Общая эффективность", f"{overall_effectiveness:.1f}%")
            
            with col2:
                st.metric("Успешных кандидатов", successful_candidates)
            
            with col3:
                st.metric("Общее количество", total_candidates)
            
            # Анализ по должностям
            if 'Worklist' in df.columns:
                st.subheader("💼 Эффективность по должностям")
                
                position_effectiveness = {}
                for position in df['Worklist'].unique():
                    if pd.notna(position):
                        position_df = df[df['Worklist'] == position]
                        position_total = len(position_df)
                        position_successful = len(position_df[position_df[status_col].isin(success_statuses)])
                        effectiveness = (position_successful / position_total) * 100 if position_total > 0 else 0
                        position_effectiveness[position] = {
                            'total': position_total,
                            'successful': position_successful,
                            'effectiveness': effectiveness
                        }
                
                # Сортируем по эффективности
                sorted_positions = sorted(position_effectiveness.items(), 
                                       key=lambda x: x[1]['effectiveness'], reverse=True)
                
                # Создаем DataFrame для отображения
                effectiveness_df = pd.DataFrame([
                    {
                        'Должность': pos,
                        'Всего кандидатов': data['total'],
                        'Успешных': data['successful'],
                        'Эффективность (%)': round(data['effectiveness'], 1)
                    }
                    for pos, data in sorted_positions
                ])
                
                st.dataframe(effectiveness_df, width="stretch")
                
                # График эффективности
                fig = px.bar(
                    x=[pos for pos, _ in sorted_positions],
                    y=[data['effectiveness'] for _, data in sorted_positions],
                    title="Эффективность найма по должностям (%)",
                    labels={'x': 'Должность', 'y': 'Эффективность (%)'}
                )
                st.plotly_chart(fig, width="stretch")
            
            # Анализ по рекрутерам
            if 'Recruiter' in df.columns:
                st.subheader("👥 Эффективность рекрутеров")
                
                recruiter_effectiveness = {}
                for recruiter in df['Recruiter'].unique():
                    if pd.notna(recruiter) and recruiter != "":
                        recruiter_df = df[df['Recruiter'] == recruiter]
                        recruiter_total = len(recruiter_df)
                        recruiter_successful = len(recruiter_df[recruiter_df[status_col].isin(success_statuses)])
                        effectiveness = (recruiter_successful / recruiter_total) * 100 if recruiter_total > 0 else 0
                        recruiter_effectiveness[recruiter] = {
                            'total': recruiter_total,
                            'successful': recruiter_successful,
                            'effectiveness': effectiveness
                        }
                
                # Сортируем по эффективности
                sorted_recruiters = sorted(recruiter_effectiveness.items(), 
                                        key=lambda x: x[1]['effectiveness'], reverse=True)
                
                # Показываем топ-10 рекрутеров
                top_recruiters = sorted_recruiters[:10]
                
                fig = px.bar(
                    x=[rec for rec, _ in top_recruiters],
                    y=[data['effectiveness'] for _, data in top_recruiters],
                    title="Топ-10 рекрутеров по эффективности (%)",
                    labels={'x': 'Рекрутер', 'y': 'Эффективность (%)'}
                )
                st.plotly_chart(fig, width="stretch")
                
                # Таблица эффективности
                recruiter_df = pd.DataFrame([
                    {
                        'Рекрутер': rec,
                        'Всего кандидатов': data['total'],
                        'Успешных': data['successful'],
                        'Эффективность (%)': round(data['effectiveness'], 1)
                    }
                    for rec, data in top_recruiters
                ])
                
                st.dataframe(recruiter_df, width="stretch")

# Функция для анализа трендов и паттернов
def trends_and_patterns_analysis(df):
    """Анализ трендов и паттернов в данных"""
    st.subheader("📈 Анализ трендов и паттернов")
    
    # Анализ корреляций между числовыми столбцами
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 1:
        st.subheader("🔗 Корреляции между числовыми переменными")
        
        # Вычисляем корреляции
        correlation_matrix = df[numeric_cols].corr()
        
        # Создаем тепловую карту
        fig = px.imshow(
            correlation_matrix,
            title="Корреляционная матрица числовых переменных",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        st.plotly_chart(fig, width="stretch")
        
        # Показываем сильные корреляции
        strong_correlations = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:  # Сильная корреляция
                    strong_correlations.append({
                        'Переменная 1': numeric_cols[i],
                        'Переменная 2': numeric_cols[j],
                        'Корреляция': round(corr_value, 3)
                    })
        
        if strong_correlations:
            st.write("**Сильные корреляции (>0.5):**")
            st.dataframe(pd.DataFrame(strong_correlations), width="stretch")
    
    # Анализ распределений
    st.subheader("📊 Анализ распределений")
    
    # Выбираем столбец для анализа
    if 'Score' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Гистограмма оценок
            fig = px.histogram(
                df, 
                x='Score', 
                title="Распределение оценок кандидатов",
                nbins=20
            )
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            # Box plot оценок
            fig = px.box(
                df, 
                y='Score', 
                title="Распределение оценок (box plot)"
            )
            st.plotly_chart(fig, width="stretch")
    
    # Анализ по группам
    if 'Worklist' in df.columns and 'State' in df.columns:
        st.subheader("🏢 Анализ по должностям и штатам")
        
        # Создаем сводную таблицу
        pivot_table = df.groupby(['Worklist', 'State']).size().unstack(fill_value=0)
        
        # Показываем топ-5 штатов для каждой должности
        st.write("**Топ-5 штатов для каждой должности:**")
        
        for position in df['Worklist'].unique():
            if pd.notna(position):
                position_data = pivot_table.loc[position].sort_values(ascending=False).head(5)
                
                fig = px.bar(
                    x=position_data.values,
                    y=position_data.index,
                    title=f"Топ-5 штатов для {position}",
                    orientation='h'
                )
                st.plotly_chart(fig, width="stretch")

# Функция для создания дашборда
def create_dashboard(df):
    """Создает информативный дашборд с ключевыми метриками"""
    st.subheader("📊 Дашборд ключевых показателей")
    
    # Основные метрики
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_candidates = len(df)
        st.metric("Всего кандидатов", f"{total_candidates:,}")
    
    with col2:
        if 'Status' in df.columns:
            active_candidates = len(df[df['Status'].str.contains('Active|Approved', case=False, na=False)])
            st.metric("Активных/Принятых", active_candidates)
        else:
            st.metric("Активных/Принятых", "N/A")
    
    with col3:
        if 'State' in df.columns:
            unique_states = df['State'].nunique()
            st.metric("Уникальных штатов", unique_states)
        else:
            st.metric("Уникальных штатов", "N/A")
    
    with col4:
        if 'Worklist' in df.columns:
            unique_positions = df['Worklist'].nunique()
            st.metric("Уникальных должностей", unique_positions)
        else:
            st.metric("Уникальных должностей", "N/A")
    
    # Дополнительные метрики
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'Recruiter' in df.columns:
            unique_recruiters = df['Recruiter'].nunique()
            st.metric("Уникальных рекрутеров", unique_recruiters)
        else:
            st.metric("Уникальных рекрутеров", "N/A")
    
    with col2:
        if 'Last App Date' in df.columns:
            try:
                df['Last App Date'] = pd.to_datetime(df['Last App Date'], errors='coerce')
                date_range = df['Last App Date'].max() - df['Last App Date'].min()
                st.metric("Диапазон дат", f"{date_range.days} дней")
            except:
                st.metric("Диапазон дат", "N/A")
        else:
            st.metric("Диапазон дат", "N/A")
    
    with col3:
        missing_data = df.isnull().sum().sum()
        st.metric("Пропущенных значений", f"{missing_data:,}")
    
    with col4:
        memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Размер данных", f"{memory_usage:.1f} МБ")
    
    # Быстрые инсайты
    st.subheader("💡 Быстрые инсайты")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Status' in df.columns:
            st.write("**Топ-5 статусов:**")
            top_statuses = df['Status'].value_counts().head(5)
            for status, count in top_statuses.items():
                percentage = (count / len(df)) * 100
                st.write(f"• {status}: {count:,} ({percentage:.1f}%)")
    
    with col2:
        if 'Worklist' in df.columns:
            st.write("**Топ-5 должностей:**")
            top_positions = df['Worklist'].value_counts().head(5)
            for position, count in top_positions.items():
                percentage = (count / len(df)) * 100
                st.write(f"• {position}: {count:,} ({percentage:.1f}%)")

# Основной интерфейс
def main():
    st.sidebar.title("📁 Данные")
    
    # Автоматическая загрузка встроенных данных
    df = load_builtin_data()
    
    if df is not None:
        st.success(f"✅ Данные автоматически загружены! Размер: {df.shape[0]} строк × {df.shape[1]} столбцов")
        
        # Информация о загруженных данных
        st.info(f"""
        📊 **Загруженные данные:**
        - **Файл:** merge-csv.com__68b9ee302f5dd.csv
        - **Записей:** {df.shape[0]:,}
        - **Столбцов:** {df.shape[1]}
        - **Период:** 2013-2025
        - **Тип:** Данные о найме сотрудников
        """)
        
        # Навигация по разделам
        st.sidebar.markdown("---")
        st.sidebar.title("📊 Разделы анализа")
        
        page = st.sidebar.radio(
            "Выберите раздел:",
            ["Дашборд", "Общий обзор", "Детальный анализ найма", "Эффективность найма", "Расширенный анализ", "Тренды и паттерны", "Продолжительность работы", "Машинное обучение"]
        )
        
        if page == "Дашборд":
            create_dashboard(df)
        
        elif page == "Общий обзор":
            analyze_data(df)
        
        elif page == "Детальный анализ найма":
            detailed_hiring_analysis(df)
        
        elif page == "Эффективность найма":
            hiring_effectiveness_analysis(df)
        
        elif page == "Расширенный анализ":
            advanced_data_analysis(df)
        
        elif page == "Тренды и паттерны":
            trends_and_patterns_analysis(df)
        
        elif page == "Продолжительность работы":
            analyze_tenure(df)
        
        elif page == "Машинное обучение":
            build_ml_model(df)
        
        # Дополнительная информация
        st.sidebar.markdown("---")
        st.sidebar.title("ℹ️ Информация")
        st.sidebar.info("""
        Это приложение анализирует данные о найме сотрудников:
        - Общая статистика данных
        - Детальный анализ найма
        - Продолжительность работы
        - Предсказательные модели
        """)
        
        # Возможность загрузить дополнительные данные
        st.sidebar.markdown("---")
        st.sidebar.title("📤 Дополнительные данные")
        uploaded_file = st.sidebar.file_uploader(
            "Или загрузите свой CSV файл",
            type=['csv'],
            help="Загрузите дополнительный CSV файл для сравнения"
        )
        
        if uploaded_file is not None:
            st.info("📤 Дополнительный файл загружен. Используйте его для сравнения с основными данными.")
    
    else:
        st.error("❌ Не удалось загрузить встроенные данные")
        st.info("👆 Попробуйте загрузить CSV файл вручную")
        
        # Загрузка файла вручную как fallback
        uploaded_file = st.sidebar.file_uploader(
            "Выберите CSV файл",
            type=['csv'],
            help="Загрузите CSV файл с данными о найме сотрудников"
        )
        
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            if df is not None:
                st.success(f"✅ Файл загружен! Размер: {df.shape[0]} строк × {df.shape[1]} столбцов")
                
                # Навигация по разделам
                st.sidebar.markdown("---")
                st.sidebar.title("📊 Разделы анализа")
                
                page = st.sidebar.radio(
                    "Выберите раздел:",
                    ["Дашборд", "Общий обзор", "Детальный анализ найма", "Эффективность найма", "Расширенный анализ", "Тренды и паттерны", "Продолжительность работы", "Машинное обучение"]
                )
                
                if page == "Дашборд":
                    create_dashboard(df)
                
                elif page == "Общий обзор":
                    analyze_data(df)
                
                elif page == "Детальный анализ найма":
                    detailed_hiring_analysis(df)
                
                elif page == "Эффективность найма":
                    hiring_effectiveness_analysis(df)
                
                elif page == "Расширенный анализ":
                    advanced_data_analysis(df)
                
                elif page == "Тренды и паттерны":
                    trends_and_patterns_analysis(df)
                
                elif page == "Продолжительность работы":
                    analyze_tenure(df)
                
                elif page == "Машинное обучение":
                    build_ml_model(df)

if __name__ == "__main__":
    main()
