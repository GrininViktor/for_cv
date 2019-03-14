# -*- coding: utf-8 -*-

# Импортируем библиотеки. Используем модели word2vec и fastText реализованные в gensim
import pymorphy2
import numpy as np
import re
import pandas as pd
import scipy.stats as st
from gensim.models import KeyedVectors
from gensim.models import FastText
from gensim.test.utils import get_tmpfile
from scipy.spatial.distance import cosine
from sklearn.decomposition import TruncatedSVD
import os.path
import pickle

morph = pymorphy2.MorphAnalyzer()


class CoursesSearchClass:
    # класс, который по запросу пользователя предлагает 5 наиболее подходящих ему внешних курсов
    # при инициализации ему передаётся excel файл c описанием имеющихся курсов и два файла с предобученными моделями
    def __init__(self, courses_xlsx_file, model_w2v_file, model_fasttext_file):

        self._path_to_w2v = model_w2v_file + '.pcl'
        self._path_to_fasttext = model_fasttext_file + '.pcl'

        # загружаем данные и модели
        self._data = pd.read_excel(courses_xlsx_file)
        self._data = self._data.dropna()
        self._data.index = range(self._data.shape[0])

        # для ускорения процесса прогружаем модели только в первый раз
        # в дальнейшем используем готовый дамп
        # загружаем модель word2vec
        if os.path.exists(self._path_to_w2v):
            with open(self._path_to_w2v, 'rb') as f:
                self._w2v = pickle.load(f)
        else:
            self._w2v = KeyedVectors.load_word2vec_format(model_w2v_file)
            with open(self._path_to_w2v, 'wb') as f:
                pickle.dump(self._w2v, f)

        # загружаем модель fasttext
        if os.path.exists(self._path_to_fasttext):
            with open(self._path_to_fasttext, 'rb') as f:
                self._fastText = pickle.load(f)
        else:
            self._fastText = FastText.load(model_fasttext_file)
            with open(self._path_to_fasttext, 'wb') as f:
                pickle.dump(self._fastText, f)

        self._modelIndicator = 1
        self._w2v_pc = None
        self._ft_pc = None
        self._vectors_of_words = None

        # преобразуем описания в токены в нормальной формы.
        # получаем векторные представления для описаний, как взвешенную сумму векторов слов
        # вычитаем из полученных векторов "главную" комп. svd разложения

        # для word2vec модели
        tks = []
        for text in self._data['Описание курса']:
            tks += self.tokens_in_text_m1(text.lower())
        self._frequencyM1 = {val: tks.count(val) for val in np.unique(tks)}
        self._w2v_Embeddings = self.run_sif_benchmark(self._data['Описание курса'], self._frequencyM1)

        # для fastText модели
        self._modelIndicator = 2
        tks = []
        for text in self._data['Описание курса']:
            tks += self.tokens_in_text_m2(text.lower())
        self._frequencyM2 = {val: tks.count(val) for val in np.unique(tks)}
        self._ft_Embeddings = self.run_sif_benchmark(self._data['Описание курса'], self._frequencyM2)

    # извлечение токенов из описания для w2v модели
    def tokens_in_text_m1(self, text):
        tokens = re.findall(r"[\w']+", text)
        allowable_tokens = []
        for token in tokens:
            tag = morph.parse(token)[0].tag.POS
            normal_form = morph.parse(token)[0].normal_form
            if tag is None:
                if normal_form + '_X' in self._w2v.vocab:
                    allowable_tokens += [normal_form + '_X']
                continue
            tag = str(tag)
            if tag == 'INFN' and (normal_form + '_VERB' in self._w2v.vocab):
                allowable_tokens += [normal_form + '_VERB']
                continue
            if tag == 'INTJ' and (normal_form + '_NOUN' in self._w2v.vocab):
                allowable_tokens += [normal_form + '_NOUN']
                continue
            if ('ADJ' in tag) and (normal_form + '_ADJ' in self._w2v.vocab):
                allowable_tokens += [normal_form + '_ADJ']
                continue
            if ((tag == 'NOUN') or (tag == 'VERB') or (tag == 'ADVB')) and ((normal_form + '_' + str(tag)) in self._w2v.vocab):
                allowable_tokens += [normal_form + '_' + tag]
        return allowable_tokens

    # извлечение токенов из описания для fastText модели
    def tokens_in_text_m2(self, text):
        tokens = re.findall(r"[\w']+", text)
        allowable_tokens = []
        for token in tokens:
            tag = morph.parse(token)[0].tag.POS
            normal_form = morph.parse(token)[0].normal_form
            if not (normal_form in self._fastText.wv.vocab):
                continue
            if tag is None:
                allowable_tokens += [normal_form]
                continue
            tag = str(tag)
            if (tag == 'INFN') or (str(tag) == 'INTJ') or (tag == 'NOUN') or ('ADJ' in tag) or (tag == 'VERB') or \
                                                                                               (tag == 'ADVB'):
                allowable_tokens += [normal_form]
                continue
        return allowable_tokens

    # вычитание из текста svd компоненты(компоненту полученную из описания курсов храним для вычитания из запроса)
    def remove_first_principal_component(self, X):
        if self._modelIndicator != 0:
            svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
            svd.fit(X)
            pc = svd.components_
            if self._modelIndicator == 1:
                self._w2v_pc = pc
            if self._modelIndicator == 2:
                self._ft_pc = pc
        if self._modelIndicator == 11:
            pc = self._w2v_pc
        if self._modelIndicator == 12:
            pc = self._ft_pc
        XX = X - X.dot(pc.transpose()) * pc
        return XX

    # sif embedding
    def run_sif_benchmark(self, sentences, freqs={}, a=0.1):
        total_freq = sum(freqs.values())
        embeddings = []
        model = self._w2v
        for sent in list(sentences):
            if self._modelIndicator == 2 or self._modelIndicator == 12:
                tokens = self.tokens_in_text_m2(sent.lower())
                if len(tokens) == 0:
                    return None
                model = self._fastText
            if self._modelIndicator == 1 or self._modelIndicator == 11:
                tokens = self.tokens_in_text_m1(sent.lower())
                if len(tokens) == 0:
                    return None
            weights = [a / (a + freqs.get(token, 0) / total_freq) for token in tokens]
            self._vectors_of_words = [model[token] for token in tokens]
            embedding = np.average(self._vectors_of_words, axis=0, weights=weights)
            embeddings.append(embedding)
        if self._modelIndicator != 0:
            embeddings = self.remove_first_principal_component(np.array(embeddings))
        return embeddings

    def crucial_word(self):
        pc = self._ft_pc
        for vec in self._vectors_of_words:
            dist = []
            for emb in self._ft_Embeddings:
                dist += [cosine(emb, vec - vec.dot(pc.transpose())*pc)]
            dist = np.array(dist)
            if st.shapiro(np.array(dist))[1] < 2e-3 and (dist < 0.647).sum() >= 2:
                return True
        return False

    # на запрос пользователя предлагает 5 курсов в пересечении рекомендованных курсов w2v и ft моделями
    def offer_courses(self, request):

        self._modelIndicator = 12

        distances2 = []
        request_embedding = self.run_sif_benchmark([request], self._frequencyM2)

        if request_embedding is None or not self.crucial_word():
            return None

        for emb in self._ft_Embeddings:
            distances2 += [cosine(emb, request_embedding)]
        distances2 = np.array(distances2)

        self._modelIndicator = 11

        distances1 = []
        request_embedding = self.run_sif_benchmark([request], self._frequencyM1)
        for emb in self._w2v_Embeddings:
            distances1 += [cosine(emb, request_embedding)]
        distances1 = np.array(distances1)

        course_numbers_m1 = []
        course_numbers_m2 = []
        while True:
            course_numbers_m1 += [np.argmin(distances1)]
            distances1[np.argmin(distances1)] = 1e8

            course_numbers_m2 += [np.argmin(distances2)]
            distances2[np.argmin(distances2)] = 1e8

            course_numbers = list(set(course_numbers_m2) & set(course_numbers_m1))
            if len(course_numbers) >= 5:
                return {
                    'name':
                        self._data.loc[course_numbers][
                            'Тематика курса'
                        ].values.tolist(),
                    'link':
                        self._data.loc[course_numbers][
                            'Ссылка на курс'
                        ].values.tolist()
                }


if __name__ == '__main__':
    cl = CoursesSearchClass(
        'data/courses_catalog.xlsx',
        'data/taiga_upos_skipgram_300_2_2018.vec',
        'data/araneum_none_fasttextskipgram_300_5_2018.model'
    )
    r = 'Как сделать красивую презентацию'
    while r != '':
        res = cl.offer_courses(r)
        print(res)
        r = input()
