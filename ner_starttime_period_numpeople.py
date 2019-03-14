# -*- coding: utf-8 -*-

# используемые библиотеки
from ner.ner_base import NerBase
import yargy
import pymorphy2
from yargy.pipelines import morph_pipeline
from IPython.display import display
from num2t4ru import num2text as n2t
from yargy import and_, not_, or_, rule, Parser
from yargy.predicates import (
    lte,
    gte,
    gram,
    custom,
    dictionary,
    normalized,
    in_
)
morph = pymorphy2.MorphAnalyzer()


class TimePeriod(object):
    def __init__(self, start_time=None, end_time=None):
        
        if start_time is not None:
            if self.TestForm(start_time):
                self._startTime = start_time
            else:
                print("StartTime must have format xx:xx")
        else:
            self._startTime = start_time
        if end_time is not None:
            if self.TestForm(end_time):
                self._endTime = end_time
            else:
                print("EndTime must have format xx:xx")
        else:
            self._endTime = end_time
    
    @staticmethod        
    def test_form(time):
        separator_ind = time.find(":")
        if separator_ind != -1:
            if(time[:separator_ind].isdigit() and (len(time[:separator_ind]) <= 2) and
                    (0 <= int(time[:separator_ind]) <= 23)) and \
                (time[separator_ind + 1:].isdigit() and
                    (len(time[separator_ind + 1:]) <= 2) and (0 <= int(time[separator_ind + 1:]) <= 59)):
                return True
            else:
                return False  
        else:
            return False

class NerTimeCount(NerBase):
    
    def __init__(self):
        super(NerTimeCount, self).__init__()
        self.last_request = None
        self.last_result = [None, None, None]
        self.name = 'TimeCount'

        # переводим числа от 1 до 59 в текст
        n2t60 = [n2t(i) for i in range(1, 60)]
        
        # для поиска порядковых числительных
        def not_coll_numbers(x):
            return ('NUMR' in str(morph.parse(x)[0].tag) and ('Coll' not in str(morph.parse(x)[0].tag))) or x == 'один'

        # часы в словах
        hours_t = and_(dictionary(n2t60[:24] + ["полтора", "полдень"]), custom(not_coll_numbers))

        # минуты в словах
        minutes_t = dictionary(n2t60)
        coll_numbers_dic = dictionary(["двое", "трое", "четверо", "пятеро", "шестеро", "семеро"])
        list_0n = {"00", "01", "02", "03", "04", "05", "06", "08", "09"}
        
        # часы в цифрах
        hours_n = or_(and_(gte(1), lte(23)), in_(list_0n))

        # минуты в цифрах
        minutes_n = or_(and_(gte(1), lte(59)), in_(list_0n))

        # разделитель в чч_мм
        two_points = dictionary([":"])
        separator = dictionary([":", "."])
        
        # определяем предлоги
        pr_v = rule("в")
        pr_ok = rule("около")
        pr_vrayone = morph_pipeline(["В районе"])
        pr_k = rule("к")
        pr_na = rule("на")
        pr_c = rule("с")
        start_prepositions = or_(
            pr_ok,
            pr_v,
            pr_k,
            pr_na,
            pr_c,
            pr_vrayone
        )
        pr_vtech = morph_pipeline(["в течение"])
        pr_do = rule("до")
        pr_po = rule("по")
        duration_prepositions = or_(
            pr_vtech,
            pr_do,
            pr_po
        )

        # отрезки времени суток
        day_periods = or_(
            rule(normalized("утро")),
            rule(normalized("день")),
            rule(normalized("вечер"))
        )
        
        # час - особый случай, т.к. сам обозначает определённое время или длительность(аналогично "человк")
        hour = rule(normalized("час"))
        people = rule(normalized("человек"))
        
        # слова перед временем начала
        start_syn = dictionary(["начало", "старт", "встреча", "переговорную", "переговорку", "пропуск"])
        start_verbs = dictionary(["начать", "прийти", "заказать", "забронировать", "выделить", "состоится"])
        
        # слова перед продолжительнотью
        duration_verbs = dictionary(["займёт", "продлится"])
        
        # слова перед временем конца
        end_verbs = dictionary(["закончить", "уйдём", "завершим"])
        end_syn = dictionary(["конец", "окончание", "завершение"])

        # для поиска времени начала, которое выделяется с помощью : или -
        start_with_separator = or_(
            rule("начало"),
            rule("старт"),
            rule("время"),
            morph_pipeline(["начало встречи"]),
            morph_pipeline(["старт встречи"]),
            morph_pipeline(["время встречи"])
        )

        duration_with_separator = or_(
            rule("продолжительность"),
            morph_pipeline(["продолжительность встречи"])
        )

        end_with_separator = or_(
            rule("конец"),
            rule("окончание"),
            rule("завершение"),
            morph_pipeline(["конец встречи"]),
            morph_pipeline(["окончание встречи"]),
            morph_pipeline(["завершение встречи"])
        )

        # относительные указатели на день(относительно сегодняшнего)
        day_pointer = or_(
            rule("понедельник"), morph_pipeline(["пн."]), rule("пн"),
            rule("вторник"), morph_pipeline(["вт."]), rule("вт"),
            rule("среда"), rule("среду"), morph_pipeline(["ср."]), rule("ср"),
            rule("четверг"), morph_pipeline(["чт."]), rule("чт"),
            rule("пятница"), rule("пятницу"), morph_pipeline(["пт."]), rule("пт"),
            rule("суббота"), rule("субботу"), morph_pipeline(["сб."]), rule("сб"),
            rule("воскресение"), rule("воскресенье"), morph_pipeline(["вс."]), rule("вс"),
            rule("завтра"), rule("послезавтра"), rule("сегодня"))

        # чужие слова
        self._foreignWords = ["этаж", "январь", "февраль", "март", "апрель", "май", "июнь",
                              "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"]

        # количественные числительные в числа
        self._Counts = {"человек": 1, "1": 1, "один": 1,
                        "два": 2, "2": 2, "двое": 2, "вдвоём": 2,
                        "трое": 3, "три": 3, "3": 3, "втроём": 3,
                        "четверо": 4, "четыре": 4, "4": 4, "вчетвером": 4,
                        "5": 5, "пять": 5, "пятеро": 5, "впятером": 5,
                        "6": 6, "шесть": 6, "шестеро": 6,
                        "7": 7, "семь": 7, "семеро": 7,
                        "8": 7, "восемь": 8,
                        "9": 7, "девять": 9,
                        "10": 7, "десять": 10}

        # приведение времени к номальной форме
        self._ToNormalHours = {"08.00": "08:00", "8": "08:00",  "восемь": "08:00",
                               "09.00": "09:00", "9": "09:00",  "девять": "09:00",
                               "10.00": "10:00", "10": "10:00", "десять": "10:00",
                               "11.00": "11:00", "11": "11:00", "одиннадцать": "11:00",
                               "12.00": "12:00", "12": "12:00", "двенадцать": "12:00", "полдень": "12:00",
                               "13.00": "13:00", "1": "13:00", "13": "13:00", "один": "13:00", "час": "13:00",
                                                                                               "часу": "13:00",
                               "14.00": "14:00", "2": "14:00", "14": "14:00", "два": "14:00",
                               "15.00": "15:00", "3": "15:00", "15": "15:00", "три": "15:00",
                               "16.00": "16:00", "4": "16:00", "16": "16:00", "четыре": "16:00",
                               "17.00": "17:00", "5": "17:00", "17": "17:00", "пять": "17:00",
                               "18.00": "18:00", "6": "18:00", "18": "18:00", "шесть": "18:00",
                               "19.00": "19:00", "7": "19:00", "19": "19:00", "семь": "19:00"}

        # приведение промежутка времени к нормальной форме
        self._ToNormalDelta = {"1": "01:00",   "один": "01:00", "час": "01:00",
                               "1:5": "01:30", "полтора": "01:30",
                               "2": "02:00",   "два": "02:00",
                               "3": "03:00",   "три": "03:00",
                               "4": "04:00",   "четыре": "04:00",
                               "5": "05:00",   "пять": "05:00",
                               "6": "06:00",   "шесть": "06:00",
                               "7": "7:00",    "семь": "07:00"}

        # правила для времени в формате from time to time
        self._rulesFromTO = [
            # from time to time
            rule(start_prepositions,
                 or_(hour, rule(or_(hours_t, hours_n))),
                 separator.optional(), 
                 minutes_n.optional(),
                 or_(day_periods, hour).optional(),
                 duration_prepositions,
                 or_(hours_t, hours_n),
                 separator.optional(), 
                 minutes_n.optional()),
            # чч:мм - чч:мм
            rule(hours_n,
                 separator, 
                 minutes_n,
                 "-",
                 hours_n,
                 separator, 
                 minutes_n),
            # day time to time
            rule(day_pointer,
                 rule(or_(hours_t, hours_n)),
                 separator.optional(), 
                 minutes_n.optional(),
                 or_(day_periods, hour).optional(),
                 duration_prepositions,
                 or_(hours_t, hours_n),
                 separator.optional(), 
                 minutes_n.optional())
        ]
        
        # правила для времени в формате from time on time
        self._rulesFromOn = [
            # from time on n hour
            rule(start_prepositions,
                 or_(hours_t, hours_n),
                 separator.optional(),
                 minutes_n.optional(),
                 or_(day_periods, hour).optional(),
                 pr_na, 
                 or_(hours_t, hours_n),
                 hour.optional()),
            # from time on hour
            rule(start_prepositions,
                 or_(hours_t, hours_n),
                 separator.optional(), 
                 minutes_n.optional(),
                 or_(day_periods, hour).optional(),
                 pr_na, 
                 hour)
        ]

        # правила для времени в формате on time from time
        self._rulesOnFrom = [
            # on n hour from time
            rule(pr_na, 
                 or_(hours_t, hours_n),
                 hour,
                 start_prepositions,
                 or_(hours_t, hours_n),
                 separator.optional(), 
                 minutes_n.optional(),
                 or_(day_periods, hour).optional()),
            # on hour from time
            rule(pr_na,
                 hour, 
                 start_prepositions,
                 or_(hours_t,hours_n),
                 separator.optional(), 
                 minutes_n.optional(),
                 or_(day_periods, hour).optional())
        ]

        # правила для времени в формате from time
        self._rulesFrom = [
            # day or start or start verb in time
            rule(or_(day_pointer, rule(start_syn), rule(start_verbs)),
                 start_prepositions,
                 or_(hours_t, hours_n),
                 separator.optional(), 
                 minutes_n.optional()),
            # start with separator
            rule(start_with_separator,
                 two_points,
                 or_(rule(hours_t), rule(hours_n)),
                 separator.optional(), 
                 minutes_n.optional()),
            # since time day or hour 
            rule(pr_c,
                 or_(rule(hours_t), rule(hours_n)),
                 separator.optional(), 
                 minutes_n.optional(),
                 or_(day_periods, hour)),
            # since hour
            rule(pr_c,
                 hour),
            # on n часов day
            rule(pr_na,
                 or_(hours_t, hours_n),
                 hour.optional(),
                 day_periods),
            # on час day
            rule(pr_na,
                 hour,
                 day_periods)
        ]

        # правила для времени окончания и продолжительности
        self._rulesTo = [
            # end or end verb in time
            rule(or_(end_syn, end_verbs),
                 start_prepositions,
                 or_(rule(hours_t), rule(hours_n), hour),
                 separator.optional(), 
                 minutes_n.optional()),
            # duration verb time-time
            rule(duration_verbs,
                 hours_n.optional(),
                 dictionary(["."]).optional(), 
                 minutes_n.optional(),
                 "-",
                 hours_n.optional(),
                 dictionary(["."]).optional(),
                 hour),
            # duration verb time
            rule(duration_verbs,
                 or_(hours_t, hours_n),
                 dictionary(["."]).optional(), 
                 minutes_n.optional(),
                 hour),
            # end with separation 
            rule(end_with_separator, two_points,
                 or_(rule(hours_t), rule(hours_n)),
                 separator.optional(), 
                 minutes_n.optional()),
            # duration with separation
            rule(duration_with_separator, two_points,
                 or_(rule(hours_t), rule(hours_n)),
                 separator.optional(), 
                 minutes_n.optional())
        ]

        # общие правила для начального, конечного времени и продолжительности
        self._rulesCommon = [
            # in time + hour or day period 
            rule(or_(pr_v, pr_vrayone,pr_k),
                 or_(hours_t, hours_n),
                 or_(hour, day_periods)),
            # on time + day period
            rule(pr_na,
                 or_(hours_t, hours_n),
                 or_(day_periods)),
            # in hh:mm
            rule(pr_v,
                 hours_n,
                 separator, 
                 minutes_n),
            # hh:mm
            rule(hours_n,
                 two_points,
                 minutes_n,
                 or_(day_periods, hour).optional()),
            # on n hour
            rule(pr_na,
                 or_(hours_t, hours_n),
                 hour),
            # on hour
            rule(pr_na,
                 hour)
        ]

        # правила для количества людей
        self._rulesCount = [
            # coll number
            rule(coll_numbers_dic),
            # n people
            rule(or_(hours_t, hours_n).optional())
        ]

        # правила используемые в повторных запросах
        self._rulesTime = [
            # всевозможные форматы времени
            rule(or_(rule(hours_t), rule(hours_n), hour),
                 separator.optional(), 
                 minutes_n.optional())
        ]
        self._rulesPeriod = [
            # всевозможные интервалы времени
            rule(or_(rule(hours_t), rule(hours_n), hour),
                 dictionary(["."]).optional(), 
                 minutes_n.optional())
        ]
        self._rulesCountPeople = [
            # количественные числительные
            rule(coll_numbers_dic),
            # n человек
            rule(or_(hours_t, hours_n).optional(), people)
        ]
        
    @staticmethod
    # поиск числительного после start preposition
    def ind_after_preposition(words):
        for i in range(len(words)):
            if words[i] in ["в", "к", "на", "с", "до", "по", "около"]:
                if words[i + 1] in ["районе", "течение"]:
                    return i + 2
                else:
                    return i + 1             

    @staticmethod
    # разность конечного и начального времени
    def delta_time(start_time, end_time):
        separator_in_start = start_time.find(":")
        separator_in_end = end_time.find(":")
        sth = int(start_time[:separator_in_start])
        stm = int(start_time[separator_in_start+1:])
        endh = int(end_time[:separator_in_end])
        endm = int(end_time[separator_in_end+1:])
        deltam = endh*60 + endm - sth*60 - stm
        return  (len(str(int(deltam/ 60))) == 1)*"0" + str(int(deltam/60)) + ":" \
                               + (len(str(deltam % 60)) == 1)*"0" + str(deltam % 60)

    # приведение времени к нормальной форме
    def time_to_normal_form(self, time):
        time = morph.parse(time)[0].normal_form
        if not TimePeriod().test_form(time):
                return self._ToNormalHours[time]
        else:
            return time 

    @staticmethod
    # проверяем последовательность символов после выделенных токенов
    def good_continuation(text):
        if len(text) < 2:
            return True
        if text[0] in ['.', ':'] and text[1].isdigit():
            return False
        else:
            return True

    # извлечение из выделенных правилами токенов конкретного времени
    def tokens_to_normal_time(self, rule, tokens):
        
        normal_form = self.time_to_normal_form
        text_time = self.text[tokens[0].span[0]:tokens[-1].span[1]]
        if not self.good_continuation(self.text[tokens[-1].span[1]:]):
            return [None, None, None]

        if text_time[-1] == '.':
            text_time = text_time[:len(text_time) - 1]
        text_time = text_time.replace(".", ":")
        text_time_split = text_time.split()
        
        if rule == self._rulesFromTO[0] or rule == self._rulesFromTO[2]:
            return [normal_form(text_time.split()[1]), self.delta_time(normal_form(text_time.split()[1]), normal_form(text_time.split()[3]))]
        if rule == self._rulesFromTO[1]:
            return [normal_form(text_time[:5]), self.delta_time(normal_form(text_time[:5]), normal_form(text_time[6:]))]
        if rule == self._rulesFromOn[0]:
            ind1 = self.ind_after_preposition(text_time_split)
            ind2 = self.ind_after_preposition(text_time_split[ind1:]) + ind1
            return [normal_form(text_time_split[ind1]), self._ToNormalDelta[text_time_split[ind2]]]
        if rule == self._rulesFromOn[1]:
            ind = self.ind_after_preposition(text_time_split)
            return [normal_form(text_time_split[ind]),  "01:00"]
        if rule == self._rulesOnFrom[0]:
            ind1 = self.ind_after_preposition(text_time_split)
            ind2 = self.ind_after_preposition(text_time_split[ind1:]) + ind1
            return [normal_form(text_time_split[ind2]), self._ToNormalDelta[text_time_split[ind1]]]
        if rule == self._rulesOnFrom[1]:
            ind1 = self.ind_after_preposition(text_time_split)
            ind2 = self.ind_after_preposition(text_time_split[ind1:]) + ind1
            return [normal_form(text_time_split[ind2]), "01:00"]
        if rule == self._rulesFrom[0]:
            if(len(self.text[tokens[-1].span[1]:].split()) == 0) or \
                    not(morph.parse(self.text[tokens[-1].span[1]:].split()[0])[0].normal_form in self._foreignWords):
                self.text = self.text[:tokens[0].span[0]] + len(self.text[tokens[0].span[0]:tokens[-1].span[1]])*" " + \
                            self.text[tokens[-1].span[1]:]
                return [normal_form(text_time_split[2]), None]
            else:
                return [None, None]   
        if rule == self._rulesFrom[1]:
            self.text = self.text[:tokens[0].span[0]] + len(self.text[tokens[0].span[0]:tokens[-1].span[1]])*" " + \
                        self.text[tokens[-1].span[1]:]
            return [normal_form(text_time_split[-1]), None]
        if rule == self._rulesFrom[2]:
            self.text = self.text[:tokens[0].span[0]] + len(self.text[tokens[0].span[0]:tokens[-1].span[1]])*" " + \
                        self.text[tokens[-1].span[1]:]
            return [normal_form(text_time_split[1]), None]
        if rule == self._rulesFrom[3]:
            self.text = self.text[:tokens[0].span[0]] + len(self.text[tokens[0].span[0]:tokens[-1].span[1]])*" " + \
                        self.text[tokens[-1].span[1]:]
            return ["13:00", None]
        if rule == self._rulesFrom[4]:
            self.text = self.text[:tokens[0].span[0]] + len(self.text[tokens[0].span[0]:tokens[-1].span[1]])*" " + \
                        self.text[tokens[-1].span[1]:]
            return [normal_form(text_time_split[1]), None]
        if rule == self._rulesFrom[5]:
            self.text = self.text[:tokens[0].span[0]] + len(self.text[tokens[0].span[0]:tokens[-1].span[1]])*" " + \
                        self.text[tokens[-1].span[1]:]
            return ["13:00", None]
        if rule == self._rulesTo[0]:
            self.text = self.text[:tokens[0].span[0]] + len(self.text[tokens[0].span[0]:tokens[-1].span[1]])*" " + \
                        self.text[tokens[-1].span[1]:]
            return [None, self.delta_time(self.times[0], normal_form(text_time_split[2]))]
        if rule == self._rulesTo[1]:
            self.text = self.text[:tokens[0].span[0]] + len(self.text[tokens[0].span[0]:tokens[-1].span[1]])*" " + \
                        self.text[tokens[-1].span[1]:]
            return [None, self._ToNormalDelta[text_time_split[1].split('-')[0]]]
        if rule == self._rulesTo[2]:
            self.text = self.text[:tokens[0].span[0]] + len(self.text[tokens[0].span[0]:tokens[-1].span[1]])*" " + \
                        self.text[tokens[-1].span[1]:]
            return [None, self._ToNormalDelta[text_time_split[1]]]
        if rule == self._rulesTo[3]:
            self.text = self.text[:tokens[0].span[0]] + len(self.text[tokens[0].span[0]:tokens[-1].span[1]])*" " + \
                        self.text[tokens[-1].span[1]:]
            return [None, self.delta_time(self.times[0],normal_form(text_time_split[-1]))]
        if rule == self._rulesTo[4]:
            self.text = self.text[:tokens[0].span[0]] + len(self.text[tokens[0].span[0]:tokens[-1].span[1]])*" " + \
                        self.text[tokens[-1].span[1]:]
            if self.times[0] is not None:
                return [None, self._ToNormalDelta[text_time_split[-1]]]
            else:
                return [None, None]
        if rule == self._rulesCommon[0]:
            self.text = self.text[:tokens[0].span[0]] + len(self.text[tokens[0].span[0]:tokens[-1].span[1]])*" " + \
                        self.text[tokens[-1].span[1]:]
            ind = self.ind_after_preposition(text_time_split)
            if not self._startExtracted:
                return [normal_form(text_time_split[ind]), None]
            else:
                return [None, normal_form(text_time_split[ind])]
        if rule == self._rulesCommon[1]:
            self.text = self.text[:tokens[0].span[0]] + len(self.text[tokens[0].span[0]:tokens[-1].span[1]])*" " + \
                        self.text[tokens[-1].span[1]:]
            if not self._startExtracted:
                return [normal_form(text_time_split[1]), None]
            else:
                return [None, normal_form(text_time_split[1])]
        if rule == self._rulesCommon[2]:
            self.text = self.text[:tokens[0].span[0]] + len(self.text[tokens[0].span[0]:tokens[-1].span[1]])*" " + \
                        self.text[tokens[-1].span[1]:]
            if not self._startExtracted:
                return [normal_form(text_time_split[1]), None]
            else:
                return [None, self.delta_time(self.times[0], normal_form(text_time_split[1]))]
        if rule == self._rulesCommon[3]:
            self.text = self.text[:tokens[0].span[0]] + len(self.text[tokens[0].span[0]:tokens[-1].span[1]])*" " + \
                        self.text[tokens[-1].span[1]:]
            if not self._startExtracted:
                return [normal_form(text_time_split[0]), None]
            else:
                return [None, self.delta_time(self.times[0], normal_form(text_time_split[0]))]
        if rule == self._rulesCommon[4]:
            self.text = self.text[:tokens[0].span[0]] + len(self.text[tokens[0].span[0]:tokens[-1].span[1]])*" " + \
                        self.text[tokens[-1].span[1]:]
            if not self._startExtracted:
                return [normal_form(text_time_split[1]), None]
            else:
                return [None, self._ToNormalDelta[text_time_split[1]]]
        if rule == self._rulesCommon[5]:
            self.text = self.text[:tokens[0].span[0]] + len(self.text[tokens[0].span[0]:tokens[-1].span[1]])*" " + \
                        self.text[tokens[-1].span[1]:]
            if not self._startExtracted:
                return ["13:00", None]
            else:
                return [None, "01:00"]

        if rule in self._rulesCountPeople + self._rulesCount:
            return [None, None, self._Counts[morph.parse(text_time_split[0])[0].normal_form]]

    # выделение из сообщения по rules токенов, содержащих время
    def extract_by(self, rules):
        ners = []
        for r in rules:
            any_rule = r.named('any_rule')
            parser = Parser(any_rule)
            found = parser.findall(self.text.lower())
            for match in found:
                ners += match.tokens
                break
            if len(ners) != 0:
                for_time_update = self.tokens_to_normal_time(r, ners)
                if for_time_update[0] is not None:
                    self.times[0] = for_time_update[0]
                    self._startExtracted = True
                if for_time_update[1] is not None:
                    self.times[1] = for_time_update[1]
                    self._endExtracted = True
                if len(for_time_update) > 2:
                    self.times[2] = for_time_update[2]
                return True
        return False


    # основная функция
    def get_entities(self, text):
        # Возвращаем результат из кэша, если он есть
        if text == self.last_request and self.last_result is not [None, None, None]:
            return self.last_result

        self._startExtracted = False
        self._endExtracted = False
        self.times = [None, None, None]
        self.text = text

        self.extract_by(self._rulesFromTO + self._rulesFromOn + self._rulesOnFrom)

        if not self._startExtracted:
            self.extract_by(self._rulesFrom)

        if not self._endExtracted:
            self.extract_by(self._rulesTo)

        while (not self._startExtracted or not self._endExtracted) and self.extract_by(self._rulesCommon):
            continue

        self.extract_by(self._rulesCountPeople)

        # Кэшируем результаты для последнего вопроса
        self.last_request = text
        self.last_result = self.times
        return self.times
    
    def has_entity(self, message):
        return self.get_entities(message) != [None, None, None]



class NerTime(NerTimeCount):
    def __init__(self):
        super(NerTime, self).__init__()
        self.last_request = None
        self.last_result = [None, None, None]
        self.name = 'Time'

    def tokens_to_normal_time(self, rule, tokens):
        # Преобразует найденные токены в начальное время
        normal_form = self.time_to_normal_form
        text_time = self.text[tokens[0].span[0]:tokens[-1].span[1]]
        if text_time[-1] == '.':
            text_time = text_time[:len(text_time) - 1]
        text_time = text_time.replace(".", ":")
        if rule == self._rulesTime[0]:
            return [normal_form(text_time), None]

    def get_entities(self, message):
        # Возвращаем результат из кэша, если он есть
        if message == self.last_request and self.last_result[0] is not None:
            return self.last_result

        self.times = [None, None, None]
        self.text = message

        self.extract_by(self._rulesTime)

        # Кэшируем результаты для последнего вопроса
        self.last_request = message
        self.last_result[0] = self.times[0]

        if self.times[0] is not None:
            return [self.times[0]]
        else:
            return []

    def has_entity(self, message):
        return len(self.get_entities(message)) > 0


class NerPeriod(NerTimeCount):
    def __init__(self, ner_time_count = None):
        super(NerPeriod, self).__init__()
        self.last_request = None
        self.last_result = [None, None, None]
        self.name = 'Period'

    def tokens_to_normal_time(self, rule, tokens):

        text_time = self.text[tokens[0].span[0]:tokens[-1].span[1]]
        if text_time[-1] == ".":
            text_time = text_time[:len(text_time) - 1]
        text_time = text_time.replace(".", ":")

        if rule == self._rulesPeriod[0]:
            return [None, self._ToNormalDelta[text_time]]

    def get_entities(self, message):
        # Возвращаем результат из кэша, если он есть
        if message == self.last_request and self.last_result[1] is not None:
            return self.last_result

        self.times = [None,None,None]
        self.text = message

        self.extract_by(self._rulesPeriod)

        # Кэшируем результаты для последнего вопроса
        self.last_request = message
        self.last_result[1] = self.times[1]

        if self.times[1] is not None:
            return [self.times[1]]
        else:
            return []

    def has_entity(self, message):
        return len(self.get_entities(message)) > 0


class NerCount(NerTimeCount):
    def __init__(self, ner_time_count = None):
        super(NerCount, self).__init__()
        self.last_request = None
        self.last_result = [None, None, None]
        self.name = 'Count'

        if ner_time_count is not None:
            self.last_request = ner_time_count.last_request
            self.last_result = ner_time_count.last_result

    def get_entities(self, message):
        # Возвращаем результат из кэша, если он есть
        if message == self.last_request and self.last_result[2] is not None:
            return self.last_result

        self.times = self.last_result
        self.text = message

        self.extract_by(self._rulesCount)

        # Кэшируем результаты для последнего вопроса
        self.last_request = message
        self.last_result[2] = self.times[2]

        if self.times[2] is not None:
            return [self.times[2]]
        else:
            return []

    def has_entity(self, message):
        return len(self.get_entities(message)) > 0

# main для проверки, что всё работает

if __name__ == '__main__':

    ner = NerTimeCount()
    print(ner.get_entities("Нужен пропуск на троих человек в 11.00.".lower()))
    
    ner_time = NerTime(ner)
    print(ner_time.get_entities("Нужен пропуск на троих человек в "
                                "12.00.".lower()))
    ner_count = NerCount(ner)
    print(ner_count.get_entities("Нужен пропуск на 4 человек".lower()))
    
    end_ner = NerPeriod(ner)
    print(end_ner.get_entities("на 2 часа"))
