from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pandas as pd 

le = LabelEncoder()

csv = pd.read_csv('data.csv', sep=';')
csv['Tipo'] = le.fit_transform(csv['Tipo'])

data = csv.values

atributtes     = data[:, 0:5]
numberComments = data[:, 5]
likes          = data[:, 6]
shared         = data[:, 7]

modelComments = LinearRegression()
modelComments.fit(atributtes, numberComments )

modelLike = LinearRegression()
modelLike.fit(atributtes, likes)

modelShared = LinearRegression()
modelShared.fit(atributtes, shared)

tipo  = int(input('Informe o número de tipo da postagem: Foto[0] | Link[1] | Status[2] | Video[3]: '))
mes   = int(input('Mês: '))
dia   = int(input('Dia da semana: D[1] | S[2] | T[3] | Q[4] | Q[5] | S[6] | S[7]: '))
hora  = int(input('Hora: '))
pago  = int(input('Pago: SIM[1] | NÃO[0]: '))

valuesComments = modelComments.predict([
    [tipo, mes, dia, hora, pago]
])

valuesLikes = modelLike.predict([
    [tipo, mes, dia, hora, pago]
])

valuesShared = modelShared.predict([
    [tipo, mes, dia, hora, pago]
])

print('Média de comentários: ', int(valuesComments)
)

print('Média de likes: ', int(valuesLikes)
)

print('Média de compartilhamentos: ', int(valuesShared)
)