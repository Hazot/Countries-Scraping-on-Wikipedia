import pandas as pd
import numpy as np
from statistics import mode
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
import itertools
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from pomegranate import BernoulliDistribution, DiscreteDistribution, ConditionalProbabilityTable
from pomegranate import Node, BayesianNetwork, NaiveBayes
import json

# Puisque ceci n'est pas corrigé, voici le plus RAW fichier possible depuis un notebook qu'on a fait

# Question 2

# ============================================================
# 1.a

fileA = pd.read_csv("./data/data_scraped.csv")
nom_col = list(fileA.columns)
array_dataA = np.array(fileA)
fileA = pd.read_csv("./data/data_scraped.csv")

nom_col = list(fileA.columns)
# nom_col

# ============================================================
# 1.b

# Initiate the lists
means = []
mediannes = []
maximums = []
modes = []
minimums = []
variances = []
nb_manquantes = []

# Calculates the things
for i in range(1, 41):
    col = array_dataA[:, i]
    col = col.astype("float")
    means.append(np.nanmean(col, dtype="float"))
    mediannes.append(np.nanmedian(col))
    maximums.append(np.nanmax(col))
    modes = mode(col)
    minimums.append(np.nanmin(col))
    variances.append(np.nanvar(col))
    nb_manquantes.append(len([x for x in col if pd.isna(x)]))

# ============================================================
# 1.c

# Enlever ligne avec valeur manquante
array_moins_douze = []
for row in array_dataA:
    nb_nan = 0
    for i in row:
        if pd.isna(i):
            nb_nan += 1
    if (nb_nan < 12):
        array_moins_douze.append(row)
array_moins_douze_avec_pays = np.array(array_moins_douze)
array_moins_douze = np.delete(np.array(array_moins_douze), 0, axis=1)

# np.array(array_moins_douze)
# df_dataA = pd.DataFrame(array_moins_douze_avec_pays)
# df_dataA.columns = nom_col
# df_dataA.to_csv("./data/dataA.csv", index=False)

# ============================================================
# 1.d

# indice qui indique ou sont les nan pour la regression plus bas
ou_sont_les_nan = []
for row in array_moins_douze:
    ligne_index_nan = []
    for i in range(0,40):
        if (pd.isna(row[i])):
            ligne_index_nan.append(i)
    ou_sont_les_nan.append(ligne_index_nan)

# ou_sont_les_nan

# Initié une list qui va devenir un array avec dataA rempli avec les médiannes associées
array_avec_medianne = []

# Créer l'array de données remplis avec les médianes
for i in range(0, 40):
    col_new = []
    col = array_moins_douze[:, i]
    medianne = mediannes[i]
    for x in col:
        if pd.isna(x):
            col_new.append(medianne)
        else:
            col_new.append(x)
    array_avec_medianne.append(col_new)
array_avec_medianne = np.transpose(array_avec_medianne)

# array_avec_medianne[0]
# mediannes

# Stock les prédictions de la 1ere régression
prediction_reg1 = []
for i in range(0,40):
    y = np.delete(array_avec_medianne[:,i],ou_sont_les_nan[i])
    x = np.delete(array_avec_medianne,i,1)
    x_moins = np.delete(x,ou_sont_les_nan[i],0)
    reg = LinearRegression().fit(x_moins,y)
    predictions = reg.predict(x)
    prediction_reg1.append(predictions)

prediction_reg1 = np.transpose(np.array(prediction_reg1))

# prediction_reg1
# ou_sont_les_nan[0]

# Initie l'array qui va stocké les données de la 2e régression faite sur les données remplies avec la médiane
array_reg1 = []
for i in range(0,len(array_avec_medianne[:,0])):
    ligne = array_avec_medianne[i]
    nan = ou_sont_les_nan[i]
    for j in nan:
        ligne[j] = prediction_reg1[i][j]
    array_reg1.append(ligne)

array_reg1 = np.array(array_reg1)

# array_reg1[:,0]
# array_reg1[0]
# array_avec_medianne[0]

# Stock les prédictions de la 2e régression
prediction_reg2 = []
for i in range(0,40):
    y = np.delete(array_reg1[:,i],ou_sont_les_nan[i])
    x = np.delete(array_reg1,i,1)
    x_moins = np.delete(x,ou_sont_les_nan[i],0)
    reg = LinearRegression().fit(x_moins,y)
    predictions = reg.predict(x)
    prediction_reg2.append(predictions)

prediction_reg2 = np.transpose(np.array(prediction_reg2))

# Initie l'array qui va stocké les données de la 2e régression faite sur les données résultantes de la 1ere régression
array_reg2 = []
for i in range(0,len(array_reg1[:,0])):
    ligne = array_reg1[i]
    nan = ou_sont_les_nan[i]
    for j in nan:
        ligne[j] = prediction_reg2[i][j]
    array_reg2.append(ligne)

array_reg2 = np.array(array_reg2)

# array_reg2
# print(array_reg2[:,0])
# print(array_reg1[:,0])
# print(array_avec_medianne[:,26])
# mediannes[26]

# ============================================================
# 1.e

# Question 1 e)
array_bin = []
for i in range(0, 40):
    col = array_reg2[:,i]
    medianne = mediannes[i]
    col_new = []
    for x in col:
        if x <=medianne:
            col_new.append(0)
        else:
            col_new.append(1)
    array_bin.append(col_new)

array_bin = np.transpose(np.array(array_bin))

# dataB
array_dataB = np.append(array_moins_douze_avec_pays[:, 0, np.newaxis], array_reg2, axis=1)
df_dataB = pd.DataFrame(array_dataB)

# print(array_dataB)

df_dataB.columns = nom_col

# df_dataB

# df_dataB.to_csv("./data/dataB.csv", index=False)

# dataC
array_dataC = np.append(array_moins_douze_avec_pays[:, 0, np.newaxis], array_bin, axis=1)
df_dataC = pd.DataFrame(array_dataC)

# array_dataC

df_dataC.columns = nom_col

# df_dataC

# df_dataC.to_csv("./data/dataC.csv", index=False)

# ============================================================
# 2.a
df_corr = pd.DataFrame(array_reg2)

correlations = df_corr.astype("float").corr()

# ============================================================
# 2.b

# Question 2
df_corr = pd.DataFrame(array_reg2)

# df_corr

correlations = df_corr.astype("float").corr()

# Inspiré de https://stackoverflow.com/questions/29432629/plot-correlation-matrix-using-pandas
# fig = plt.figure(figsize=(19,15))
# plt.matshow(correlations, fignum=fig.number)
# plt.xticks(range(df_corr.select_dtypes(['number']).shape[1]), df_corr.select_dtypes(['number']).columns, fontsize=14, rotation=45)
# plt.yticks(range(df_corr.select_dtypes(['number']).shape[1]), df_corr.select_dtypes(['number']).columns, fontsize=14)
# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=14)
# plt.title("Matrice de corrélation", fontsize=16);

# fig.savefig("correlations.pdf", bbox_inches='tight')

# fig = plt.figure(figsize=(8,4))
# plt.scatter(np.arange(40), nb_manquantes)
# plt.title("Nb de données manquantes en fonction de la colonne")
# plt.ylabel("Nb de données manquantes")
# plt.xlabel("# de la colonne")
# plt.show()

# fig.savefig("données_manquantes.pdf", bbox_inches='tight')

array_correlation = np.array(correlations)
# corr_json = array_correlation.tolist()
# with open('./data/corr.json', 'w') as f:
#     json.dump(corr_json, f)

# 2.b
correlations_max = []
for i in range(40):
    col = np.abs(array_correlation[:, i])
    col[i] = 0
    corr_max = np.max(col)
    argmax = np.argmax(col)
    correlations_max.append(argmax)

# max_corr_json = np.array(correlations_max).tolist()
# with open('./data/max_corr.json', 'w') as f:
#     json.dump(max_corr_json, f)

# 2.c
correlation_moyennes = []
for i in range(40):
    col = np.abs(array_correlation[:, i])
    correlation_moyennes.append([i, np.mean(col)])


# correlation_moyennes

# permet de ne compter que sur la 2e composante, soit la 2e colonne pour faire un action (ex: sort sur la 2e fonction)
def deuxieme_composante(array):
    return (array[1])


correlation_moyennes.sort(reverse=True, key=deuxieme_composante)

# correlation_moyennes

# ordre_json = np.array(correlation_moyennes)[:,0].tolist()
# with open('./data/ordre.json', 'w') as f:
#     json.dump(ordre_json, f)


# ============================================================
# 3.a

df_array_reg2 = pd.DataFrame(np.copy(array_reg2))

for col in df_array_reg2.columns:
    col_zscore = col
    df_array_reg2[col_zscore] = (df_array_reg2[col] - df_array_reg2[col].mean()) / df_array_reg2[col].std(ddof=0)

# df_array_reg2

array_reg2_norma = df_array_reg2.to_numpy()

scores_avec_toutes_les_colonnes = []
for i in range(40):
    Y = array_reg2_norma[:, i]
    X = np.delete(array_reg2_norma, i, 1)
    reg = LinearRegression().fit(X, Y)
    scores_avec_toutes_les_colonnes.append([i, reg.score(X, Y)])

arrays_scores_vs_all_others = np.array(scores_avec_toutes_les_colonnes)
# arrays_scores_vs_all_others

# x = arrays_scores_vs_all_others[:, 0]
# y = arrays_scores_vs_all_others[:, 1]
# score_median = np.median(arrays_scores_vs_all_others[:, 1])
# fig, ax = plt.subplots(figsize=(10,10))
# ax.scatter(x, y)
# ax.axhline(score_median, color='r')
# plt.title("Scores du prédicteur d'une variable en fonctions de toutes les autres variables (Médiane en rouge)")
# plt.ylabel("Scores du prédicteur en fonctions des autres colonnes")
# plt.xlabel("Colonne à prédire")
# plt.show()

# fig.savefig("reg_scores_plot.pdf", bbox_inches='tight')

sorted_arrays_scores_vs_all_others = sorted(arrays_scores_vs_all_others, reverse=True, key=deuxieme_composante)
# sorted_arrays_scores_vs_all_others

# plus difficile à prédire
# sorted_arrays_scores_vs_all_others[-1]

# plus facile à prédire
# sorted_arrays_scores_vs_all_others[0]

# 2.b
test = list(itertools.combinations(range(40), 2))

Y_test = array_reg2_norma[:, 0]
indices = np.delete(np.array(range(40)), 0)
# Paires triangulaires supérieurs
paires_test = list(itertools.combinations(indices, 2))

scores_test = []
for x, y in paires_test:
    #     print(x,y)
    X_paires = np.transpose(np.array([array_reg2_norma[:, x], array_reg2_norma[:, y]]))
    #     print(X_paires.shape)
    reg_test = LinearRegression().fit(X_paires, Y_test)
    scores_test.append([[x, y], reg_test.score(X_paires, Y_test)])

# scores_test

scores_test.sort(reverse=True, key=deuxieme_composante)

# scores_test[0]

meilleures_paires_regression = []
tout_paires = []
for i in range(40):
    Y = array_reg2_norma[:, i]
    indices = np.delete(np.array(range(40)), i)
    paires = list(itertools.combinations(indices, 2))
    scores_test = []
    for x, y in paires:
        X_paires = np.transpose(np.array([array_reg2_norma[:, x], array_reg2_norma[:, y]]))
        reg_test = LinearRegression().fit(X_paires, Y)
        scores_test.append([[x, y], reg_test.score(X_paires, Y)])
    scores_test.sort(reverse=True, key=deuxieme_composante)
    meilleures_paires_regression.append(scores_test[0])

# meilleures_paires_regression[1][0]

# lineaire_paires_json = [row[0] for row in meilleures_paires_regression]
# lineaire_paires_json = np.array(lineaire_paires_json).tolist()
# with open('./data/lineaire_paires_colonnes1.json', 'w') as f:
#     json.dump(lineaire_paires_json, f)

nb_occurences_paires = []
for i in range(40):
    nb_occurences_paires.append(meilleures_paires_regression[i][0][0])
    nb_occurences_paires.append(meilleures_paires_regression[i][0][1])

nb_occurences = np.unique(np.array(nb_occurences_paires), return_counts=True)

nb_occurences_list = []

for i, x in enumerate(nb_occurences[0]):
    nb_occurences_list.append([x, nb_occurences[1][i]])

nb_occurences_per_column = sorted(nb_occurences_list, reverse=True, key=deuxieme_composante)

# fig = plt.figure(figsize=(8,5))
# plt.scatter(np.arange(len(nb_occurences_per_column)), nb_occurences[1])
# plt.ylabel("Nb d'occurences d'une colonne parmi les paires")
# plt.xlabel("# de la colonne")
# plt.title("Nb d'occurences parmi les paires en fonction de la colonne")
# plt.show()

# fig.savefig("lineaire_nb_occurences_paires.pdf", bbox_inches='tight')

# 3.c
indices = np.array(range(40))

paires = list(itertools.combinations(indices, 2))

moyenne_scores = []
for x, y in paires:
    tout_les_scores = []
    y_possible = np.delete(indices, [x, y])
    for label in y_possible:
        Y = array_reg2_norma[:, label]
        X_paires = np.transpose(np.array([array_reg2_norma[:, x], array_reg2_norma[:, y]]))
        reg_test = LinearRegression().fit(X_paires, Y)
        score = reg_test.score(X_paires, Y)
        tout_les_scores.append(score),

    moyenne_scores.append([[x, y], np.mean(tout_les_scores)])

# moyenne_scores

moyenne_scores.sort(reverse=True, key=deuxieme_composante)

# moyenne_scores

# moyenne_scores[0]

# nom_col[25]

# Question 3 bayes
# 3.a
scores_avec_toutes_les_colonnes_bayes = []
for i in range(40):
    Y = array_bin[:,i]
    X = np.delete(array_bin,i,1)
    GNB = GaussianNB().fit(X, Y)
    scores_avec_toutes_les_colonnes_bayes.append([i,GNB.score(X,Y)])

arrays_scores_vs_all_others_bayes = np.array(scores_avec_toutes_les_colonnes_bayes)
# arrays_scores_vs_all_others_bayes

# x = arrays_scores_vs_all_others_bayes[:, 0]
# y = arrays_scores_vs_all_others_bayes[:, 1]
# score_median = np.mean(arrays_scores_vs_all_others_bayes[:, 1])
# fig, ax = plt.subplots(figsize=(10,10))
# ax.scatter(x, y)
# ax.axhline(score_median, color='r')
# plt.title("Scores du prédicteur bayésien d'une variable en fonctions de toutes les autres variables (Médiane en rouge)")
# plt.ylabel("Scores du prédicteur en fonctions des autres colonnes")
# plt.xlabel("Colonne à prédire")
# plt.show()

# fig.savefig("bayes_scores_plot.pdf", bbox_inches='tight')

# 3.a (ensemble)

# labels = ["Reg", "Bayes"]
# x_bayes = arrays_scores_vs_all_others_bayes[:, 0]
# y_bayes = arrays_scores_vs_all_others_bayes[:, 1]
# x_reg = arrays_scores_vs_all_others[:, 0]
# y_reg = arrays_scores_vs_all_others[:, 1]
# fig, ax = plt.subplots(figsize=(10,10))
# ax.scatter(x_reg, y_reg, label=labels[0])
# ax.scatter(x_bayes, y_bayes, label=labels[1])
# plt.title("Scores du prédicteur d'une variable en fonctions de toutes les autres variables (Reg et Bayes)")
# plt.ylabel("Scores du prédicteur en fonctions des autres colonnes")
# plt.xlabel("Colonne à prédire")
# plt.legend(labels)
# plt.show()

# fig.savefig("reg_and_bayes_scores_plot.pdf", bbox_inches='tight')

# 3.b
scores_test_bayes = []
meilleures_paires_bayes = []
tout_paires = []
for i in range(40):
#     print(i)
    Y = array_bin[:,i]
    indices = np.delete(np.array(range(40)), i)
    paires =  list(itertools.combinations(indices, 2))
    scores_test = []
    for x,y in paires:
        X_paires = np.transpose(np.array([array_bin[:,x], array_bin[:,y]]))
        bayes_test = GaussianNB().fit(X_paires,Y)
        accuracy = accuracy_score(Y,bayes_test.predict(X_paires))
        scores_test.append([[x,y],accuracy])
        scores_test_bayes.append([[x,y],accuracy])
    scores_test = sorted(scores_test, reverse=True,key=deuxieme_composante)
    meilleures_paires_bayes.append(scores_test[0])

# scores_test_bayes

# meilleures_paires_bayes

moyenne_scores_bayes = []
for x,y in paires:
    #print(x,y)
    tout_les_scores= []
    y_possible = np.delete(indices,[x,y])
#     print(y_possible)
    for label in y_possible:
        Y = array_bin[:,label]
        X_paires = np.transpose(np.array([array_bin[:,x], array_bin[:,y]]))
        bayes_test = GaussianNB().fit(X_paires,Y)
        score = accuracy_score(Y,bayes_test.predict(X_paires))
        tout_les_scores.append(score)
    moyenne_scores_bayes.append([[x,y],np.mean(tout_les_scores)])

# moyenne_scores_bayes

moyenne_scores_bayes.sort(reverse=True,key=deuxieme_composante)

# moyenne_scores_bayes

# moyenne_scores_bayes[0]

# bayesien_paires_json = [row[0] for row in meilleures_paires_regression]
# bayesien_paires_json = np.array(bayesien_paires_json).tolist()
# with open('./data/bayesien_paires_colonnes.json', 'w') as f:
#     json.dump(bayesien_paires_json, f)

# smth_bayes, y_reg = zip(*moyenne_scores)
# smth_bayes, y_bayes = zip(*moyenne_scores_bayes)
# labels = ["Reg", "Bayes"]
# x_q3 = np.arange(100)
# fig, ax = plt.subplots(figsize=(10,10))
# ax.scatter(x_q3, y_reg[:100], label=labels[0])
# ax.scatter(x_q3, y_bayes[:100], label=labels[1])
# plt.title("100 meilleurs scores moyens des prédicteur (Reg et Bayes)")
# plt.ylabel("Scores du prédicteur en fonctions des autres colonnes")
# plt.xlabel("Colonne à prédire")
# plt.legend(labels)
# plt.show()

# fig.savefig("reg_and_bayes_scores_moyens_plot.pdf", bbox_inches='tight')


# 4.a
pca = PCA(n_components=2)
pca.fit(array_reg2)
deux_dim = pca.fit_transform(array_reg2)

# deux_dim

# fig, ax = plt.subplots(figsize=(25,25))
# ax.scatter(deux_dim[:,0], deux_dim[:,1])
# n = array_moins_douze_avec_pays[:,0]

# for i, txt in enumerate(n):
#     ax.annotate(txt, (deux_dim[:,0][i], deux_dim[:,1][i]))

# fig.savefig("pca_25x25.pdf", bbox_inches='tight')

pca_norma = PCA(n_components=2)
pca_norma.fit(array_reg2_norma)
deux_dim_norma = pca_norma.fit_transform(array_reg2_norma)

# fig, ax = plt.subplots(figsize=(25,25))
# ax.scatter(deux_dim_norma[:,0], deux_dim_norma[:,1])
# n = array_moins_douze_avec_pays[:,0]

# for i, txt in enumerate(n):
#     ax.annotate(txt, (deux_dim_norma[:,0][i], deux_dim_norma[:,1][i]))

# fig.savefig("pca_norma_25x25.pdf", bbox_inches='tight')

iso = Isomap(n_components=2, n_neighbors=5)
x_transformed = iso.fit_transform(array_reg2_norma)

# fig, ax = plt.subplots(figsize=(25,25))
# ax.scatter(x_transformed[:,0], x_transformed[:,1])
# n = array_moins_douze_avec_pays[:,0]

# for i, txt in enumerate(n):
#     ax.annotate(txt, (x_transformed[:,0][i], x_transformed[:,1][i]))

# fig.savefig("isomap_norma_25x25.pdf", bbox_inches='tight')

# ANALYSE PCA ===================================

# 4.b
pca_5 = PCA(n_components=5)
pca_5.fit(array_reg2)
x_cinq_dim = pca_5.fit_transform(array_reg2)

# Analyse PCA (non norma)
x_pca_deux = deux_dim
r_deux = []
r_cinq = []
for i in range(40):
    Y = array_reg2[:,i]
    reg_2 = LinearRegression().fit(x_pca_deux,Y)
    reg_5 = LinearRegression().fit(x_cinq_dim,Y)
    r_deux.append(reg_2.score(x_pca_deux,Y))
    r_cinq.append(reg_5.score(x_cinq_dim,Y))

# r_deux

# r_cinq

r_cinq.sort(reverse=True)
r_deux.sort(reverse=True)

# print(r_cinq)

# print(r_deux)

# print(np.mean(r_deux))
# print(np.mean(r_cinq))

# Analyse norma
pca_5_norma = PCA(n_components=5)
pca_5_norma.fit(array_reg2_norma)
x_cinq_dim_norma = pca_5_norma.fit_transform(array_reg2_norma)

# Analyse PCA norma
x_pca_deux_norma = deux_dim_norma
r_deux_norma = []
r_cinq_norma = []
for i in range(40):
    Y = array_reg2_norma[:,i]
    reg_2_norma = LinearRegression().fit(x_pca_deux_norma,Y)
    reg_5_norma = LinearRegression().fit(x_cinq_dim_norma,Y)
    r_deux_norma.append(reg_2_norma.score(x_pca_deux_norma,Y))
    r_cinq_norma.append(reg_5_norma.score(x_cinq_dim_norma,Y))

r_deux_norma_indexed = []
for i, r in enumerate(r_deux_norma):
    r_deux_norma_indexed.append([i, r])

r_deux_norma_sorted_indexed = sorted(r_deux_norma_indexed, reverse=True, key=deuxieme_composante)

# Stocker l'ordre de PCA 2
pca_order_analysis = []
for i in (r_deux_norma_sorted_indexed):
    pca_order_analysis.append(i[0])

# Créer le vecteur PCA 2 dans cet ordre
r_deux_norma_in_order = []
for i in pca_order_analysis:
    r_deux_norma_in_order.append(r_deux_norma[i])

# Créer le vecteur PCA 5 dans cet ordre
r_cinq_norma_in_order = []
for i in pca_order_analysis:
    r_cinq_norma_in_order.append(r_cinq_norma[i])

# labels = ["PCA_2", "PCA_5"]
# fig = plt.figure(figsize=(10,10))
# plt.scatter(np.arange(40), r_deux_norma_in_order, label=labels[0])
# plt.ylabel("Score du prédicteur en fonctions des autres colonnes en ordre")
# plt.xlabel("Colonne dans l'ordre des meilleurs score pour PCA 2")
# plt.title("Score du prédicteur pour une même colonne avec PCA 2 et 5")
# plt.scatter(np.arange(40), r_cinq_norma_in_order, label=labels[1])
# plt.legend(labels)
# plt.show()

# fig.savefig("pca_analysis.pdf", bbox_inches='tight')

# print(np.mean(r_deux_norma_in_order))
# print(np.mean(r_cinq_norma_in_order))


# 4.a
pca = PCA(n_components=2)
pca.fit(array_reg2)
deux_dim = pca.fit_transform(array_reg2)

# deux_dim

# fig, ax = plt.subplots(figsize=(25,25))
# ax.scatter(deux_dim[:,0], deux_dim[:,1])
# n = array_moins_douze_avec_pays[:,0]

# for i, txt in enumerate(n):
#     ax.annotate(txt, (deux_dim[:,0][i], deux_dim[:,1][i]))

# fig.savefig("pca_25x25.pdf", bbox_inches='tight')

pca_norma = PCA(n_components=2)
pca_norma.fit(array_reg2_norma)
deux_dim_norma = pca_norma.fit_transform(array_reg2_norma)

# fig, ax = plt.subplots(figsize=(25,25))
# ax.scatter(deux_dim_norma[:,0], deux_dim_norma[:,1])
# n = array_moins_douze_avec_pays[:,0]

# for i, txt in enumerate(n):
#     ax.annotate(txt, (deux_dim_norma[:,0][i], deux_dim_norma[:,1][i]))

# fig.savefig("pca_norma_25x25.pdf", bbox_inches='tight')

iso = Isomap(n_components=2, n_neighbors=5)
x_transformed = iso.fit_transform(array_reg2_norma)

# fig, ax = plt.subplots(figsize=(25,25))
# ax.scatter(x_transformed[:,0], x_transformed[:,1])
# n = array_moins_douze_avec_pays[:,0]

# for i, txt in enumerate(n):
#     ax.annotate(txt, (x_transformed[:,0][i], x_transformed[:,1][i]))

# fig.savefig("isomap_norma_25x25.pdf", bbox_inches='tight')

# ANALYSE PCA ===================================

# 4.b
pca_5 = PCA(n_components=5)
pca_5.fit(array_reg2)
x_cinq_dim = pca_5.fit_transform(array_reg2)

# Analyse PCA (non norma)
x_pca_deux = deux_dim
r_deux = []
r_cinq = []
for i in range(40):
    Y = array_reg2[:,i]
    reg_2 = LinearRegression().fit(x_pca_deux,Y)
    reg_5 = LinearRegression().fit(x_cinq_dim,Y)
    r_deux.append(reg_2.score(x_pca_deux,Y))
    r_cinq.append(reg_5.score(x_cinq_dim,Y))

# r_deux

# r_cinq

r_cinq.sort(reverse=True)
r_deux.sort(reverse=True)

# print(r_cinq)

# print(r_deux)

# print(np.mean(r_deux))
# print(np.mean(r_cinq))

# Analyse norma
pca_5_norma = PCA(n_components=5)
pca_5_norma.fit(array_reg2_norma)
x_cinq_dim_norma = pca_5_norma.fit_transform(array_reg2_norma)

# Analyse PCA norma
x_pca_deux_norma = deux_dim_norma
r_deux_norma = []
r_cinq_norma = []
for i in range(40):
    Y = array_reg2_norma[:,i]
    reg_2_norma = LinearRegression().fit(x_pca_deux_norma,Y)
    reg_5_norma = LinearRegression().fit(x_cinq_dim_norma,Y)
    r_deux_norma.append(reg_2_norma.score(x_pca_deux_norma,Y))
    r_cinq_norma.append(reg_5_norma.score(x_cinq_dim_norma,Y))

r_deux_norma_indexed = []
for i, r in enumerate(r_deux_norma):
    r_deux_norma_indexed.append([i, r])

r_deux_norma_sorted_indexed = sorted(r_deux_norma_indexed, reverse=True, key=deuxieme_composante)

# Stocker l'ordre de PCA 2
pca_order_analysis = []
for i in (r_deux_norma_sorted_indexed):
    pca_order_analysis.append(i[0])

# Créer le vecteur PCA 2 dans cet ordre
r_deux_norma_in_order = []
for i in pca_order_analysis:
    r_deux_norma_in_order.append(r_deux_norma[i])

# Créer le vecteur PCA 5 dans cet ordre
r_cinq_norma_in_order = []
for i in pca_order_analysis:
    r_cinq_norma_in_order.append(r_cinq_norma[i])

# labels = ["PCA_2", "PCA_5"]
# fig = plt.figure(figsize=(10,10))
# plt.scatter(np.arange(40), r_deux_norma_in_order, label=labels[0])
# plt.ylabel("Score du prédicteur en fonctions des autres colonnes en ordre")
# plt.xlabel("Colonne dans l'ordre des meilleurs score pour PCA 2")
# plt.title("Score du prédicteur pour une même colonne avec PCA 2 et 5")
# plt.scatter(np.arange(40), r_cinq_norma_in_order, label=labels[1])
# plt.legend(labels)
# plt.show()

# fig.savefig("pca_analysis.pdf", bbox_inches='tight')

# print(np.mean(r_deux_norma_in_order))
# print(np.mean(r_cinq_norma_in_order))

# 5.a
nombre_de_fleches = 100
np.fill_diagonal(array_correlation, 0)

corr_abs = np.abs(array_correlation)

indices = np.array(range(40))
paires = list(itertools.combinations(indices, 2))

# len(paires)

corr_en_vecteur = []
for x, y in paires:
    corr_en_vecteur.append([[x, y], corr_abs[x][y]])

# corr_en_vecteur

corr_en_vecteur.sort(reverse=True, key=deuxieme_composante)

# corr_en_vecteur[0:nombre_de_fleches]

correlation_moyennes.sort()

table_node_1 = []

dict_corr_moyenne = {x[0]: x[1] for x in correlation_moyennes}

corr_max_4_parents = []
dict_parent = {i: 0 for i in range(40)}
dict_parent_node = {i: [] for i in range(40)}
enfants = []
for x in corr_en_vecteur:
    pair_current = x[0]
    if dict_corr_moyenne[pair_current[0]] >= dict_corr_moyenne[pair_current[1]]:
        parent = pair_current[0]
        enfant = pair_current[1]
    else:
        parent = pair_current[1]
        enfant = pair_current[0]

    nombre_de_parent_current = dict_parent[enfant]

    if nombre_de_parent_current < 4:
        corr_max_4_parents.append(x)
        enfants.append(enfant)
        dict_parent[enfant] += 1
        dict_parent_node[enfant].append(parent)
    if len(corr_max_4_parents) == nombre_de_fleches:
        break

test = []
test.append(array_bin[:, 26])
test.append(array_bin[:, 0])
test.append(array_bin[:, 22])
test = np.transpose(test)

test_prop_cond = np.concatenate(
    (array_bin[:, 26, np.newaxis], array_bin[:, 0, np.newaxis], array_bin[:, 22, np.newaxis]), axis=1)

cond = np.unique(test_prop_cond, axis=0, return_counts=True)[1] / len(test_prop_cond)

cond_tous_nodes = []
tout_states = []
for i in range(40):
    parents = dict_parent_node[i]
    states_possibles = []
    states_possibles.append(array_bin[:, i])
    for x in parents:
        states_possibles.append(array_bin[:, x])
    states_possibles = np.transpose(states_possibles)
    cond = np.unique(states_possibles, axis=0, return_counts=True)[1] / len(states_possibles)
    cond_tous_nodes.append(cond)

    states_append = np.unique(states_possibles, axis=0)
    states_append = np.concatenate((states_append, cond[:, np.newaxis]), axis=1)
    tout_states.append(states_append)
dict_cond = {i: cond_tous_nodes[i] for i in range(len(cond_tous_nodes))}
dict_state = {i: tout_states[i] for i in range(len(tout_states))}

for i in range(40):
    state_sans_proba = np.delete(dict_state[i], -1, 1)
    # print(state_sans_proba)
    state_qui_devraient_etre_la = np.array(list(itertools.product([0, 1], repeat=len(state_sans_proba[0]))))
    # print("allo", state_qui_devraient_etre_la)
    nombre_ajouter = 0
    for k in range(len(state_qui_devraient_etre_la)):
        x = state_qui_devraient_etre_la[k]
        dedans = False
        for j in state_sans_proba:
            if np.array_equal(x, j):
                dedans = True
        if dedans == False:
            nombre_ajouter += 1
            dict_state[i] = np.insert(dict_state[i], k * (len(state_sans_proba[0]) + 1), np.append(x, 0))
            dict_state[i] = dict_state[i].reshape(len(state_sans_proba[:, 0]) + nombre_ajouter,
                                                  len(state_sans_proba[0]) + 1)
    dict_state[i] = dict_state[i].reshape(len(state_qui_devraient_etre_la[:, 0]), len(state_sans_proba[0]) + 1)


def bin_composante(array):
    bits = np.delete(array, -1, 1)
    bits = bits.astype("int")
    str_bits = []
    for bit in bits:
        str_bits.append("".join(map(str, bit)))
    #     print(str_bits[10])
    #     print(int(str_bits[10], 2))
    return (int(str_bits[0], 2))


# init dict distribution
dict_distribution = {i: 0 for i in range(40)}

# dict_distribution

for k in range(100):
    for i in range(40):
        if dict_distribution[i] == 0:
            nb_parents = dict_parent[i]
            if nb_parents == 0:
                dict_distribution[i] = DiscreteDistribution({float(j): dict_state[i][j][1] for j in range(2)})
            else:
                parents = dict_parent_node[i]
                tout_dedans = True
                for x in parents:
                    if dict_distribution[x] == 0:
                        tout_dedans = False
                if tout_dedans == True:
                    dict_distribution[i] = ConditionalProbabilityTable(list(dict_state[i]),
                                                                       [dict_distribution[j] for j in
                                                                        dict_parent_node[i]])

# dict_distribution[0]

# dict_distribution[5]

nodes = {i: Node(dict_distribution[i], str(i)) for i in range(40)}

bayesnet = BayesianNetwork("network")

for i in range(40):
    bayesnet.add_node(nodes[i])

for i in range(40):
    parents = dict_parent_node[i]
    for x in parents:
        bayesnet.add_edge(nodes[x], nodes[i])

bayesnet.bake()

# with open('./data/reseau.json', 'w') as f:
#     f.write(bayesnet.to_json())

comb = list(itertools.combinations(range(40), 2))

dict_comb_baye = {}

comb_baye = []

for x, y in comb:
    arr1 = []
    arr2 = []
    arr3 = []
    arr4 = []
    for i in range(0, x):
        arr1.append(None)
        arr2.append(None)
        arr3.append(None)
        arr4.append(None)
    arr1.append(0)
    arr2.append(0)
    arr3.append(1)
    arr4.append(1)
    for i in range(x + 1, y):
        arr1.append(None)
        arr2.append(None)
        arr3.append(None)
        arr4.append(None)
    arr1.append(0)
    arr2.append(1)
    arr3.append(0)
    arr4.append(1)
    for i in range(y, 39):
        arr1.append(None)
        arr2.append(None)
        arr3.append(None)
        arr4.append(None)
    comb_baye.append([arr1, arr2, arr3, arr4])

# print(comb_baye[779])

# print(comb_baye[0])

print("Everything is running fine!")
print("==========================")
print("End")
