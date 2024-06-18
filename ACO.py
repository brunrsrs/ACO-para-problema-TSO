#ACO para TSP
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx

#distância entre as cidades
def dist(x1, y1, x2, y2):
  return math.sqrt((x1-x2)**2 + (y1-y2)**2)

# ---------------------------------------------------

#gera possíveis soluções para depois calcular as distâncias
def calcula_solucao(e, feromonios, distancias, alfa, beta):
  # "atribui" formiga a uma cidade aleatoria
  solucao = [np.random.randint(e)]
  visitado = []
  visitado.append(solucao[0])

  for m in range(e-1):
    cidade_atual = solucao[-1]
    probabilidade = []
    for cidade in range(e):
      if cidade not in visitado:
        probabilidade.append((feromonios[cidade_atual][cidade] ** alfa) * ((1/distancias[cidade_atual][cidade]) ** beta))

      else:
        probabilidade.append(0)

    probabilidade = np.array(probabilidade)
    probabilidade = probabilidade / np.sum(probabilidade)
    proxima_cidade = random.choices(range(e), weights=probabilidade)[0]
    solucao.append(proxima_cidade)
    visitado.append(proxima_cidade)
  return solucao

# ---------------------------------------------------

#função para atualizar o feromonio pelo codigo passado em aula
def atualiza_feromonio(feromonios, solucoes, distancias, ro, Q, melhor_solucao, melhor_distancia, b, e):
  feromonios *= 1-ro
  var_feromonio = np.zeros_like(feromonios)
  for solucao in solucoes:
      Lk = 0
      for k in range(len(solucao)-1):
        Lk += distancias[solucao[k]][solucao[k+1]]
      Lk += distancias[solucao[-1]][solucao[0]]
      for i in range(len(solucao) - 1):
          var_feromonio[solucao[i]][solucao[i + 1]] += Q / Lk
          var_feromonio[solucao[i + 1]][solucao[i]] += Q / Lk
      var_feromonio[solucao[-1]][solucao[0]] += Q / Lk
      var_feromonio[solucao[0]][solucao[-1]] += Q / Lk

  Lmelhor = melhor_distancia
  for i in range(len(melhor_solucao) - 1):
      var_feromonio[melhor_solucao[i]][melhor_solucao[i + 1]] += Q / Lmelhor
      var_feromonio[melhor_solucao[i + 1]][melhor_solucao[i]] += Q / Lmelhor
  var_feromonio[melhor_solucao[-1]][melhor_solucao[0]] += Q / Lmelhor
  var_feromonio[melhor_solucao[0]][melhor_solucao[-1]] += Q / Lmelhor

  feromonios += var_feromonio + b * var_feromonio

# ---------------------------------------------------

#ACO para TSP
def ACO(max_it, alfa, beta, ro, N, e, Q, t0, b):
  rota = []
  convergencia=[]
  #separa as cidades
  cidades = []
  intervalo = [6, 6+e]
  with open('berlin52.tsp', 'r') as f:
    for linha in f.readlines()[intervalo[0]:intervalo[1]]:
      lin, x, y = map(float, linha.split())
      cidades.append([x, y])

  #calcula matriz de distâncias
  distancias = np.zeros((len(cidades), len(cidades)))
  for i in range(len(cidades)):
    for j in range(len(cidades)):
      if i != j:
        distancias[i][j] = dist(cidades[i][0], cidades[i][1], cidades[j][0], cidades[j][1])

  #atribui o feromonio inicial
  tij = np.zeros((len(cidades), len(cidades)))
  for i in range(len(cidades)):
    for j in range(len(cidades)):
      tij[i][j] = t0

  melhor_solucao = None
  melhor_distancia = float('inf')

  for i in range(max_it):
    solucoes = []
    for j in range(N):
      #adiciona a um vetor de soluçoes
      solucoes.append(calcula_solucao(e, tij, distancias, alfa, beta))

    #calcula a distância dessses vetores
    for solucao in solucoes:
      aux_dist = 0
      for k in range(len(solucao)-1):
        aux_dist += distancias[solucao[k]][solucao[k+1]]
      aux_dist += distancias[solucao[-1]][solucao[0]]

    #atualiza a melhor distância gerada
      if aux_dist < melhor_distancia:
        melhor_distancia = aux_dist
        melhor_solucao = solucao
    convergencia.append(melhor_distancia)
    atualiza_feromonio(tij, solucoes, distancias, ro, Q, melhor_solucao, melhor_distancia, b, e)

  return melhor_solucao, melhor_distancia, cidades,convergencia

# ---------------------------------------------------

# função para plotar o grafo das cidades e o caminho percorrido
def plot_graph(coordinates, solution):
    num_cities = len(coordinates)
    G = nx.Graph()

    # adiciona os nós (cidades)
    for i in range(num_cities):
        G.add_node(i, pos=coordinates[i])

    # adiciona as arestas (conexões entre cidades)
    for i in range(num_cities):
        G.add_edge(solution[i], solution[(i + 1) % num_cities])

    pos = nx.get_node_attributes(G, 'pos')

    # desenha o grafo das cidades
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, node_size=300, node_color='yellow', with_labels=True, font_size=10, font_color='black')

    # desenha o caminho percorrido
    path_edges = [(solution[i], solution[(i + 1) % num_cities]) for i in range(num_cities)]
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='black', width=2.0)

    plt.title('Grafo das cidades e caminho percorrido')
    plt.show()

# ---------------------------------------------------

alfa = 1 # alfa e beta = pesos definidos pelo usuário
beta = 5
ro = 0.5 # ro = taxa de evaporação
N = 52 # N = número de formigas
e = 52 # e = número de cidades
Q = 100 # Q = definido pelo usuario
t0 = 0.000001 # t0 = quantidade inicial de feromônio
b = 5 # b = número de formigas elitistas

melhor_solu, melhor_dist, coordenadas,convergencia = ACO(500, alfa, beta, ro, N, e, Q, t0, b)

print("Menor distância encontrada: ", melhor_dist)
print("Caminho com a menor distância: ", melhor_solu)

plot_graph(coordenadas, melhor_solu)

# plotar o gráfico dos melhores valores
plt.plot(convergencia)
plt.xlabel('Iterações')
plt.ylabel('Melhor valor encontrado')
plt.title('Distância ao longo das iterações')
plt.show()