import sqlite3
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    con = sqlite3.connect('data/archive/mental_health.sqlite')
    df = pd.read_sql_query("SELECT AnswerText, UserID, QuestionID FROM Answer WHERE UserID IN "
                   "(SELECT UserID FROM Answer WHERE QuestionID = 17"
                   ") AND (QuestionID = 17 OR QuestionID = 1)"
                   "ORDER BY UserID, QuestionID;", con)
    df_ageQuestion = df[df.QuestionID == 1]
    df_MentalHealthQuestion = df[df.QuestionID == 17]

    df_MentalHealthQuestion = df_MentalHealthQuestion.replace({
    "Somewhat easy": 1,
    "I don't know" : 2,
    "-1" : 2,
    "Very easy": 3,
    "Somewhat difficult": 4,
    "Difficult": 5,
    "Neither easy nor difficult" : 6,
    "Very difficult": 7})

    df_ageQuestion = df_ageQuestion.sort_values(by=["AnswerText"])
    df_graph = [df_ageQuestion.AnswerText.values, df_MentalHealthQuestion.AnswerText.values]
    print(df_MentalHealthQuestion.AnswerText)
    sns.scatterplot(x=df_ageQuestion.AnswerText.values, y=df_MentalHealthQuestion.AnswerText.values)
    fig = plt.gcf()
    fig.set_size_inches(12, 5)
    plt.show()
    # answerPath = 'data/results/main_Answer.xlsx'
    # questionPath = 'data/results/main_Question.xlsx'
    # df_answer = pd.read_excel(answerPath)
    # print(df_answer.UserID[df_answer.QuestionID == 1])
    # print(df_answer.UserID[df_answer.QuestionID == 17])
    # x = df_answer[df_answer.AnswerText == 17]
    # y = df_answer[df_answer.AnswerText == 1]
    # for (i=0; i < x.shape(); i++):
    # graph = sns.scatterplot(x=x, y=y)
    # graph.show()
