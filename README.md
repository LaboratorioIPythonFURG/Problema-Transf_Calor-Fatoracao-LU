# Problema

Uma consideração importante no estudo de transferência de calor é a de determinar a distribuição de
temperatura assintótica de uma placa fina quando a temperatura em seu bordo é conhecida. Suponha que a placa na
Figura 2 represente uma seção transversal de uma barra de metal, com fluxo de calor desprezível na direção
perpendicular à placa. Sejam $T_1, T_2, \dots, T_6$ as temperaturas em seis vértices interiores do reticulado da Figura 1. A temperatura num vértice é aproximadamente igual à média dos quatro vértices vizinhos mais próximos - à esquerda, acima, à direita e abaixo. Por exemplo,

$$ T_1 = \frac{(10+20+T_2+T_4)}{4} \hspace{0.5cm} \text{ou} \hspace{0.5cm} 4T_1-T_2-T_4=30  $$

| <img src="figura2.png" width="300px"></img> |
|:--:|
|*Figura 1. Temperatura em seis vértices interiores do reticulado*|

**a)** Escreva um sistema de seis equações cuja solução forneça estimativas para as temperaturas $T_1, T_2, \dots, T_6$

**b)** Resolva o sistema linear obtido em **a)** por:

    1. Eliminação Gaussiana (sem e com pivotamento)
    2. Fatoração LU (sem e com pivotamento)

# Solução

## a) Equações

$$
T_1 = \frac{1}{4}(10 + 20 + T_2 + T_4) \\
T_2 = \frac{1}{4}(T_1 + 20 + T_3 + T_5) \\
T_3 = \frac{1}{4}(T_2 + 20 + 40 + T_6) \\
T_4 = \frac{1}{4}(10 + T_1 + T_5 + 20) \\
T_5 = \frac{1}{4}(T_4 + T_2 + T_6 + 20) \\
T_6 = \frac{1}{4}(T_5 + T_3 + 40 + 20)
$$

Logo,

$$ \begin{cases}
    4T_1 &-&  T_2 &+& 0T_3 &-&  T_4 &+& 0T_5 &+& 0T_6 &=& 30 \\
    -T_1 &+& 4T_2 &-& T_3  &+& 0T_4 &-& T_5  &+& 0T_6 &=& 20 \\
    0T_1 &-&  T_2 &+& 4T_3 &+& 0T_4 &+& 0T_5 &-&  T_6 &=& 60  \\
    -T_1 &+& 0T_2 &+& 0T_3 &+& 4T_4 &-& T_5  &+& 0T_6 &=& 30 \\
    0T_1 &-&  T_2 &+& 0T_3 &-&  T_4 &+& 4T_5 &-&  T_6 &=& 20 \\
    0T_1 &+& 0T_2 &-& T_3  &+& 0T_4 &-& T_5  &+& 4T_6 &=& 60
\end{cases} $$

Que resulta na matriz de coeficientes:

$$
\begin{bmatrix}
    4 & -1 & 0 & -1 & 0 & 0 \\
    -1 & 4 & -1 & 0 & -1 & 0 \\
    0 & -1 & 4 & 0 & 0 & -1 \\
    -1 & 0 & 0 & 4 & -1 & 0 \\
    0 & -1 & 0 & -1 & 4 & -1 \\
    0 & 0 & -1 & 0 & -1 & 4
\end{bmatrix}
$$


## b.1) [Eliminação Gaussiana](https://pt.wikipedia.org/wiki/Elimina%C3%A7%C3%A3o_de_Gauss)

### Definição do Sistema

Começamos definindo o sistema que temos interesse em resolver, utilizando a identidade:

$$ Ax = b $$


```python
import numpy as np
```


```python
A = np.array([[ 4, -1,  0, -1,  0,  0], 
              [-1,  4, -1,  0, -1,  0], 
              [ 0, -1,  4,  0,  0, -1], 
              [-1,  0,  0,  4, -1,  0], 
              [ 0, -1,  0, -1,  4, -1],
              [ 0,  0, -1,  0, -1,  4]]).astype(np.float)
b = np.array([30, 20, 60, 30, 20, 60]).astype(np.float)
```


```python
m = A.shape[0]
n = A.shape[1]
print(f'Dimensões da matriz: {A.shape}')

A_expandida = np.insert(A, n, b, axis=1)
print('Matrix expandida:')
print(A_expandida)
```

    Dimensões da matriz: (6, 6)
    Matrix expandida:
    [[ 4. -1.  0. -1.  0.  0. 30.]
     [-1.  4. -1.  0. -1.  0. 20.]
     [ 0. -1.  4.  0.  0. -1. 60.]
     [-1.  0.  0.  4. -1.  0. 30.]
     [ 0. -1.  0. -1.  4. -1. 20.]
     [ 0.  0. -1.  0. -1.  4. 60.]]


### Sem pivotamento

O primeiro passo da eliminação gaussiana sem pivotamento compreende em calcular a _[matriz triangular superior](https://pt.wikipedia.org/wiki/Matriz_triangular)_ da matriz. Um algoritmo simples para esse passo envolve somar toda linha $a_i$ abaixo da linha $a_j$ por um múltiplo $\gamma$ de forma que:

$$
    a_{i,j} + \gamma a_{j,j} = 0 \Rightarrow
    a_{i} := a_{i} + \gamma a_{j} \hspace{0.25cm} \forall i > j
$$

Se $j$ varia de $1$ a $m$ numa matriz $A_{m,n}$, a matriz resultante cumpre as condições especificadas.



```python
def matriz_triangular_superior(M):
    """
        Calcula a matriz triangular superior da matriz M utilizando
        um algoritmo simples da eliminação de Gauss
    """
    
    # Criar cópia da matriz expandida
    A_new = M.copy()
    m = A_new.shape[0]
    n = A_new.shape[1]
    
    # Para cada coluna - 1 na matriz (ignoramos a coluna das variáveis dependentes)
    for j in range(n-1):
        # Selecionamos o pivô, como não há pivotamento, selecionamos os valores da diagonal
        ajj = A_new[j, j]
        
        # Para cada linha i abaixo de j
        for i in range(j+1, m):
            # Selecionamos o valor da linha i na coluna j (abaixo de ajj)
            aij = A_new[i, j]
            
            # Calculamos o múltiplo necessário do pivô ajj para que
            # x * ajj + aij = 0
            coeficiente = abs(aij) / ajj * -np.sign(aij)
            
            # Multiplicamos a linha inteira pelo coeficiente calculado
            A_i = A_new[j,:]*coeficiente
            
            # Atualizamos o valor da linha para o novo valor, que 
            # deve ser 0 na posição aij
            A_new[i] = A_new[i,:]+A_i
            
    return A_new


A_u = matriz_triangular_superior(A_expandida)
print(A_u.round(1))
```

    [[ 4.  -1.   0.  -1.   0.   0.  30. ]
     [ 0.   3.8 -1.  -0.2 -1.   0.  27.5]
     [ 0.   0.   3.7 -0.1 -0.3 -1.  67.3]
     [ 0.   0.   0.   3.7 -1.1 -0.  40.5]
     [ 0.   0.   0.   0.   3.4 -1.1 43.8]
     [ 0.   0.   0.   0.   0.   3.4 92.1]]



```python
# Aqui podemos verificar o resultado
# É possivel generalizar essa função para matrizes arbitrárias
def resolver_sistema(ms):
    T6 =  A_u[5,6] / A_u[5, 5]
    T5 = (A_u[4,6] - A_u[4,5]*T6) / A_u[4,4]
    T4 = (A_u[3,6] - A_u[3,5]*T6 - A_u[3,4]*T5) / A_u[3,3]
    T3 = (A_u[2,6] - A_u[2,5]*T6 - A_u[2,4]*T5 - A_u[2,3]*T4) / A_u[2,2]
    T2 = (A_u[1,6] - A_u[1,4]*T5 - A_u[1,3]*T4 - A_u[1,2]*T3) / A_u[1,1]
    T1 = (A_u[0,6] - A_u[0,3]*T4 - A_u[0,1]*T2) / A_u[0,0]
    return np.array([T1, T2, T3, T4, T5, T6])

# Construir vetor x com os valores encontrados
x = resolver_sistema(A_u)

# Multiplicar pela matriz original
A@x
```




    array([30., 20., 60., 30., 20., 60.])




```python
# Vamos comparar com b
display(A@x)
display(b)
```


    array([30., 20., 60., 30., 20., 60.])



    array([30., 20., 60., 30., 20., 60.])


Os dois valores são iguais, portanto, nosso algoritmo funciona.

#### Redução à forma de Gauss-Jordan
Um algoritmo para reduzir a matriz triangular a uma matriz diagonal é simples e consiste na aplicação do algoritmo anterior em ordem inversa. Isto é, ao invés descermos a matriz diagonalmente e reduzirmos os valores abaixo de $a_{j,j}$ à zero, subimos no sentido contrário e reduzimos os valores acima de $a_{j,j}$ à zero. 

$$
    a_{i,j} + \gamma a_{j,j} = 0 \Rightarrow
    a_{i} := a_{i} + \gamma a_{j} \hspace{0.25cm} \forall i < j
$$

Onde agora $j$ varia de $m$ a $1$.


```python
def matriz_triangular_para_diagonal(M):
    # Criar cópia da matriz original
    A_new = M.copy()
    
    # Dimensões da matriz
    m = A_new.shape[0]
    n = A_new.shape[1]
    
    # Iterar sobre toda coluna j da matriz, começando
    # pelo valor mais baixo da sua diagonal.
    # Aqui, a função reversed inverte o intervalo retornado
    # por range.
    for j in reversed(range(m)):
        
        # Selecionar o valor na matriz na posição (j, j),
        # ou seja, na sua diagonal
        ajj = A_new[j, j]
        
        # Multiplicar a linha por 1/ajj, para que
        # o valor em (j,j) = 1
        A_new[j] = A_new[j]*(1/ajj)
        
        # Atualizar ajj para o valor normalizado
        ajj = A_new[j,j]
        
        # Observe que o intervalo retornado pela função range é 
        # aberto em relação a seu limite superior j, assim,
        # o retorno da função reversed resulta em (j-0]
        # logo, i varia de 0 a j-1 nesse loop.
        for i in reversed(range(j)):  
        # Para cada valor no intervalo (j-0]
        
            # Selecionamos o valor aij
            aij = A_new[i, j]
            
            # Calculamos gamma, ou o coeficiente de ajj para que
            # aij + ajj*gamma = 0
            coeficiente = abs(aij) / ajj * -np.sign(aij)
            
            # aj * gamma
            A_i = A_new[j]*coeficiente
            
            # Atualizar linha i com o novo valor
            # ai + aj*gamma = 0
            A_new[i] = A_new[i]+A_i
    
    return A_new

A_d = matriz_triangular_para_diagonal(A_u)
A_d.round(2)
```




    array([[ 1.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  , 17.14],
           [ 0.  ,  1.  ,  0.  ,  0.  ,  0.  ,  0.  , 21.43],
           [ 0.  ,  0.  ,  1.  ,  0.  ,  0.  ,  0.  , 27.14],
           [ 0.  ,  0.  ,  0.  ,  1.  ,  0.  ,  0.  , 17.14],
           [ 0.  ,  0.  ,  0.  ,  0.  ,  1.  ,  0.  , 21.43],
           [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  1.  , 27.14]])



Agora podemos verificar o resultado multiplicando a última coluna da matriz diagonal ```A_d``` pela matriz original ```A```:


```python
A @ A_d[:, -1]
```




    array([30., 20., 60., 30., 20., 60.])



E o resultado, de fato, é o esperado.

Podemos então compor as duas funções que criamos para facilitar o processo.


```python
def resolver(A, b):
    # Criar matriz expandida
    A2 = np.insert(A, A.shape[1], b, axis=1)
    x = matriz_triangular_para_diagonal(matriz_triangular_superior(A2))[:, -1]
    return x

display(resolver(A, b))
display(A @ resolver(A, b))
```


    array([17.14285714, 21.42857143, 27.14285714, 17.14285714, 21.42857143,
           27.14285714])



    array([30., 20., 60., 30., 20., 60.])


### Com pivotamento

Pivotamento envolve reordenar as colunas e linhas da matriz em cada passo da eliminação Gaussiana, para que o pivô atual seja o maior valor da matriz em um passo específico.

Para facilitar o processo primeiro implementamos funções para trocar duas linhas e duas colunas de uma matriz.


```python
def trocar_linha(A, i, j):
    """
    Troca a linha i pela linha j de uma matriz A
    """
    # É importante copiar a linha, pois caso contrário
    # linha_i é uma referência à matriz original,
    # e alterar A[i] alteraria a linha_i também.
    linha_i = A[i,:].copy()
    A[i,:] = A[j,:]
    A[j,:] = linha_i
    return A

def trocar_coluna(A, i, j):
    """
    Troca a coluna i pela coluna j de uma matriz A
    """
    linha_i = A[:, i].copy()
    A[:, i] = A[:, j]
    A[:, j] = linha_i
    return A
    
matriz_teste = np.matrix([[1, 2], [3, 4]])
display(matriz_teste)
display(trocar_linha(matriz_teste, 0, 1))

display(matriz_teste)
display(trocar_coluna(matriz_teste, 0, 1))
```


    matrix([[1, 2],
            [3, 4]])



    matrix([[3, 4],
            [1, 2]])



    matrix([[3, 4],
            [1, 2]])



    matrix([[4, 3],
            [2, 1]])


> **NOTA** Observe que a variável ```matriz_teste``` utilizada como parâmetro das funções ```trocar_linha``` e ```trocar_coluna``` é alterada dentro da função, e a segunda linha ```display(matriz_teste)``` nos mostra uma matriz de formato $\begin{bmatrix} 3 & 4 \\ 1 & 2 \end{bmatrix} $. Para evitar esse comportamento, podemos passar como parâmetro uma cópia da matriz que queremos trocar as linhas/colunas: ```matriz_2 = trocar_linha(matriz_teste.copy(), 0, 1))```.

Agora podemos implementar a solução com pivotamento reutilizando a função para a eliminação Gaussiana definida antes:


```python

# Vamos testar com o sistema de equações providenciado no problema.
A2 = np.insert(A, A.shape[1], b, axis=1)

max_lin = np.max(M2[0,:-1])
max_col = np.max(M2[:-1, 0])

m = np.max(M2)
i, j = np.where(M2 == np.max(M2))

# Um cuidado necessário nessa implementação deve ser o de não considerar
# a última coluna na busca pelo maior valor da matriz, pois esta contém
# os valores de variáveis dependentes.
def maximizar_pivo(M, index=0, ignore_last_col=True, log=False, I = None):
    M_view = M[:, :-1] if ignore_last_col else M
    
    # Caso tenhamos mais de um valor maior
    v = np.argmax(M_view)
    i = int(np.floor(v/M_view.shape[1]))
    j = int(v - i*M_view.shape[1])
    
    if log:
        display(f'Coordenadas do valor maior: {i}, {j}')

    trocar_linha(M, 0, i)
    trocar_coluna(M, 0, j)
    
    # Fazer mesma alteração na matriz identidade
    if I is not None:
        trocar_linha(I, 0, i)
        trocar_coluna(I, 0, j)
    return M, I

def matriz_triangular_superior(M, pivotamento=False, log=False):
    """
        Calcula a matriz triangular superior da matriz M utilizando
        um algoritmo simples da eliminação de Gauss
    """
    # Criar cópia da matriz expandida
    A_new = M.copy()
    m = A_new.shape[0]
    n = A_new.shape[1]
    
    I = np.eye(m)
    
    # Para cada coluna - 1 na matriz (ignoramos a coluna das variáveis dependentes)
    for j in range(n-1):
        
        if log:
            display('Estado atual da matriz:')
            display(A_new.round(1))
        
        # Maximizar pivô
        if pivotamento:
            maximizar_pivo(A_new[j:, j:], j, True, log, I[j:, j:])
            
        if log:
            display('Resultado pivotamento:')
            display(A_new.round(1))
        
        # Selecionamos o pivô, como não há pivotamento, selecionamos os valores da diagonal
        ajj = A_new[j, j]
        
        # Para cada linha i abaixo de j
        for i in range(j+1, m):
            # Selecionamos o valor da linha i na coluna j (abaixo de ajj)
            aij = A_new[i, j]
            
            # Calculamos o múltiplo necessário do pivô ajj para que
            # x * ajj + aij = 0
            coeficiente = abs(aij) / ajj * -np.sign(aij)
            
            # Multiplicamos a linha inteira pelo coeficiente calculado
            A_i = A_new[j,:]*coeficiente
            
            # Atualizamos o valor da linha para o novo valor, que 
            # deve ser 0 na posição aij
            A_new[i] = A_new[i,:]+A_i
            
        if log:
            display('Resultado:')
            display(A_new.round(1))
    return A_new, I


display('Matriz original:')
display(A2.round(1))

display('Matriz escalonada sem pivotamento: ')
A_es, _ = matriz_triangular_superior(A2, False)
display(A_es.round(1))

display('Matriz escalonada com pivotamento: ')
A_ec, I = matriz_triangular_superior(A2, True)
display(A_ec.round(1))
display(I)

```


    'Matriz original:'



    array([[ 4., -1.,  0., -1.,  0.,  0., 30.],
           [-1.,  4., -1.,  0., -1.,  0., 20.],
           [ 0., -1.,  4.,  0.,  0., -1., 60.],
           [-1.,  0.,  0.,  4., -1.,  0., 30.],
           [ 0., -1.,  0., -1.,  4., -1., 20.],
           [ 0.,  0., -1.,  0., -1.,  4., 60.]])



    'Matriz escalonada sem pivotamento: '



    array([[ 4. , -1. ,  0. , -1. ,  0. ,  0. , 30. ],
           [ 0. ,  3.8, -1. , -0.2, -1. ,  0. , 27.5],
           [ 0. ,  0. ,  3.7, -0.1, -0.3, -1. , 67.3],
           [ 0. ,  0. ,  0. ,  3.7, -1.1, -0. , 40.5],
           [ 0. ,  0. ,  0. ,  0. ,  3.4, -1.1, 43.8],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  3.4, 92.1]])



    'Matriz escalonada com pivotamento: '



    array([[ 4. , -1. ,  0. , -1. ,  0. ,  0. , 30. ],
           [ 0. ,  4. , -1. ,  0. ,  0. , -1. , 60. ],
           [ 0. ,  0. ,  4. , -1. , -1. , -1. , 20. ],
           [ 0. ,  0. ,  0. ,  3.5, -0.5, -0.2, 42.5],
           [ 0. ,  0. ,  0. ,  0. ,  3.5, -0.5, 83. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  3.1, 66.3]])



    array([[1., 0., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0., 0.],
           [0., 0., 1., 0., 0., 0.],
           [0., 0., 0., 1., 0., 0.],
           [0., 0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 0., 1.]])


### Verificando resultado da matriz escalonada com pivotamento

Podemos verificar o resultado com a função ```resolver_sistema(A)``` que definimos anteriormente:


```python
x = resolver_sistema(A_ec)
display(A@x)
```


    array([17.14285714, 21.42857143, 27.14285714, 17.14285714, 21.42857143,
           27.14285714])



    array([30., 20., 60., 30., 20., 60.])



```python

def matriz_triangular_para_diagonal(M, pivotamento=False):
    # Criar cópia da matriz original
    A_new = M.copy()
    
    # Dimensões da matriz
    m = A_new.shape[0]
    n = A_new.shape[1]
    
    I = np.eye(m)
    
    # Iterar sobre toda coluna j da matriz, começando
    # pelo valor mais baixo da sua diagonal.
    # Aqui, a função reversed inverte o intervalo retornado
    # por range.
    for j in reversed(range(m)):
        
        
        # Selecionar o valor na matriz na posição (j, j),
        # ou seja, na sua diagonal
        ajj = A_new[j, j]
        
        # Multiplicar a linha por 1/ajj, para que
        # o valor em (j,j) = 1
        A_new[j] = A_new[j]*(1/ajj)
        
        # Atualizar ajj para o valor normalizado
        ajj = A_new[j,j]
        
        # Observe que o intervalo retornado pela função range é 
        # aberto em relação a seu limite superior j, assim,
        # o retorno da função reversed resulta em (j-0]
        # logo, i varia de 0 a j-1 nesse loop.
        for i in reversed(range(j)):  
        # Para cada valor no intervalo (j-0]
        
            if pivotamento and i > 1:
                maximizar_pivo(A_new[:i, :i], i, True, False, I[:i, :i])
        
            # Selecionamos o valor aij
            aij = A_new[i, j]
            
            # Calculamos gamma, ou o coeficiente de ajj para que
            # aij + ajj*gamma = 0
            coeficiente = abs(aij) / ajj * -np.sign(aij)
            
            # aj * gamma
            A_i = A_new[j]*coeficiente
            
            # Atualizar linha i com o novo valor
            # ai + aj*gamma = 0
            A_new[i] = A_new[i]+A_i
    
    return A_new, I

A_ecd, I = matriz_triangular_para_diagonal(A_ec, False)
display(A_ecd.round(1))

A @ A_ecd[:, -1:]
```


    array([[ 1. ,  0. ,  0. ,  0. ,  0. ,  0. , 18.3],
           [ 0. ,  1. ,  0. ,  0. ,  0. ,  0. , 25.7],
           [ 0. ,  0. ,  1. ,  0. ,  0. ,  0. , 21.5],
           [ 0. ,  0. ,  0. ,  1. ,  0. ,  0. , 17.6],
           [ 0. ,  0. ,  0. ,  0. ,  1. ,  0. , 27.1],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  1. , 21.4]])





    array([[30.        ],
           [35.9630102 ],
           [38.95408163],
           [24.73852041],
           [43.85204082],
           [37.04081633]])



## b.2) [Fatoração LU](https://pt.wikipedia.org/wiki/Decomposi%C3%A7%C3%A3o_LU)

### Sem pivotamento


```python
#A2 = A.copy()
#A_0 = A2.copy()
As = [A.copy()]
Ls = [np.eye(A.shape[0])]

for j in range(As[0].shape[0]):
    A_i = As[j]
    display(A_i[j, :])
    a_nn = A_i[j,j]
    
    A_j = A_i.copy()
    #L_n = Ls[j].copy()
    L_n = np.eye(A_i.shape[0])
    for i in range(j+1, n):
        a_in = A_j[i,j]
        l_in = -a_in/a_nn
        #display(A_i[i,:]*l_in+A_j[i,:])
        #A_j[i,:] = (A_i[j,:]*l_in+A_j[i,:])
        L_n[i, j] = l_in
        
    A_n = L_n@A_j
    
    As.append(A_n)
    Ls.append(L_n)

#display(Ls)
display(A)
L = np.eye(A.shape[0])
for L_i in Ls:
    L = L@np.linalg.inv(L_i)
display(L.round(1))
display((L*As[-1]).round(1))
#display((Ls[-1]@As[-1]).round())
    
```


    array([ 4., -1.,  0., -1.,  0.,  0.])



    array([ 0.  ,  3.75, -1.  , -0.25, -1.  ,  0.  ])



    array([ 0.        ,  0.        ,  3.73333333, -0.06666667, -0.26666667,
           -1.        ])



    array([ 0.        ,  0.        ,  0.        ,  3.73214286, -1.07142857,
           -0.01785714])



    array([ 0.        ,  0.        ,  0.        ,  0.        ,  3.40669856,
           -1.07655502])



    array([0.        , 0.        , 0.        , 0.        , 0.        ,
           3.39185393])



    array([[ 4., -1.,  0., -1.,  0.,  0.],
           [-1.,  4., -1.,  0., -1.,  0.],
           [ 0., -1.,  4.,  0.,  0., -1.],
           [-1.,  0.,  0.,  4., -1.,  0.],
           [ 0., -1.,  0., -1.,  4., -1.],
           [ 0.,  0., -1.,  0., -1.,  4.]])



    array([[ 1. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [-0.2,  1. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. , -0.3,  1. ,  0. ,  0. ,  0. ],
           [-0.2, -0.1, -0. ,  1. ,  0. ,  0. ],
           [ 0. , -0.3, -0.1, -0.3,  1. ,  0. ],
           [ 0. ,  0. , -0.3, -0. , -0.3,  1. ]])



    array([[ 4. , -0. ,  0. , -0. ,  0. ,  0. ],
           [-0. ,  3.8, -0. , -0. , -0. ,  0. ],
           [ 0. , -0. ,  3.7, -0. , -0. , -0. ],
           [-0. , -0. , -0. ,  3.7, -0. , -0. ],
           [ 0. , -0. , -0. , -0. ,  3.4, -0. ],
           [ 0. ,  0. , -0. , -0. , -0. ,  3.4]])




### Com pivotamento

## Solução simples (numpy.linalg.solve)

O algoritmo na biblioteca usa decomposição LU.


```python
x = np.linalg.solve(A, b)
display(x)
```


    array([17.14285714, 21.42857143, 27.14285714, 17.14285714, 21.42857143,
           27.14285714])



```python
A@x # Verificando.
```




    array([30., 20., 60., 30., 20., 60.])


