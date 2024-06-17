# Identificação de Sinais e Gestos de Libras com Redes Neurais LSTM

## Visão Geral

Este projeto visa identificar sinais e gestos da Linguagem de Sinais Brasileira (Libras) utilizando redes neurais LSTM 
(Long Short-Term Memory) e a ferramenta MediaPipe. Através deste projeto, esperamos contribuir para a acessibilidade e 
inclusão de pessoas com deficiência auditiva, facilitando a comunicação entre falantes de Libras e ouvintes.

## Linguagem de Sinais Brasileira (Libras)

Libras é a língua de sinais utilizada pela comunidade surda no Brasil. Assim como qualquer língua, ela possui sua 
própria gramática e sintaxe, sendo composta por sinais feitos com as mãos, expressões faciais e movimentos corporais. 
A Libras é reconhecida oficialmente como meio legal de comunicação e expressão no Brasil, tendo sido incluída na Lei 
de Diretrizes e Bases da Educação Nacional.

## LSTM (Long Short-Term Memory)

LSTM é um tipo de rede neural recorrente (RNN) projetada para aprender dependências de longo prazo. Diferente das 
RNNs tradicionais, as LSTMs conseguem armazenar informações por longos períodos, graças a uma 
arquitetura especial que inclui células de memória e mecanismos de controle como gates de entrada, saída e 
esquecimento. Isso as torna particularmente úteis para tarefas de série temporal e sequências, como reconhecimento 
de fala, tradução de idiomas e, neste caso, identificação de sinais de linguagem de sinais.

## Ferramentas e Tecnologias

- **Python**: A linguagem de programação utilizada para desenvolver o projeto - versão 3.8.
- **MediaPipe**: Um framework de machine learning para construir pipelines multimodais customizáveis. Utilizado aqui para captura e processamento dos movimentos das mãos, pose e rosto.
- **TensorFlow/Keras**: Utilizados para a construção e treinamento da rede neural LSTM.

## Estrutura do Projeto

1. **Captura de Dados**: Utilizando a ferramenta MediaPipe, capturamos os movimentos das mãos em tempo real, que são então convertidos em coordenadas espaciais.
2. **Processamento de Dados**: As coordenadas capturadas são processadas e formatadas para serem utilizadas no treinamento do modelo LSTM.
3. **Treinamento do Modelo**: A rede neural LSTM é treinada utilizando um conjunto de dados de sinais de Libras, permitindo que ela aprenda a reconhecer e classificar os gestos corretamente.
4. **Coleta de logs**: Durante a etapa de treinamento, logs do desempenho da rede são coletados e armazenados pela ferramenta TensorBoard.
5. **Inferência e Reconhecimento**: O modelo treinado é utilizado para identificar sinais em tempo real, traduzindo gestos de Libras para texto.

## Como Executar o Projeto

1. Clone este repositório:
   ```bash
   git clone https://github.com/DayaneCordeiro/libras-recognition-system.git
   ```
2. Navegue até o diretório do projeto:
   ```bash
   cd libras-recognition-system
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
4. Execute o script principal:
   ```bash
   python main.py
   ```

## Como executar os logs após o treinamento da rede

Para uma visualização gráfica dos logs, foi utilizada a classe TensorBoard da biblioteca 
<code>keras.src.callbacks</code>. Para execução da biblioteca é necessário seguir os seguintes passos:
1. Na pasta raiz do projeto, navegue até o diretório de logs:
   ```bash
   cd Logs/train
   ```
2. Execute o seguinte comando:
   ```bash
   tensorboard --logdir=.
   ```
3. Após a execução do comando, abra o navegador no seguinte endereço:
   ```bash
   http://localhost:6006
   ```

## Metodologia utilizada
O maior desafio encontrado no desenvolvimento projeto foi trabalhar com os gestos. Foram realizadas primeiramente duas 
implementações de uma solução satisfatória, uma trabalhando com visão computacional e outra com uma rede 
neural convolucional (CNN). Ambas não obtiveram sucesso.<br>

A terceira implementação, foi a que se apresentou um resultado mais próximo do desejado, porém ainda com muitos pontos
de melhoria que serão citados em um tópico posterior. Essa implementação utiliza uma rede neural recorrente LSTM. Como
foi citado anteriormente, esse tipo de rede possui um mecanismo de memória, que ajuda no desafio de identificar uma
sequência de sinais que formam um gesto. Tendo em vista os benefícios dessa rede, foi utilizado como base um tutorial 
de implementação da rede, disponível no seguinte endereço: [YouTube](https://www.youtube.com/watch?v=doDUihpj6ro). 
O código ensinado pelo autor, foi adaptado para atender ao caso de uso do projeto.<br>

Dada a implementação do código, utilizando a mesma arquitetura de rede neural que o autor, foi observada uma queda
muito grande nos valores da acurácia a cada vez que uma classe era adicionada ao model. Sendo que com três classes
contendo os gestos "Obrigado", "Tudo bem?" e "Saudade" a rede apresentou quase 100% de acurácia durante o treinamento,
mas ao adicionar a letra "J", a acurária caiu cerca de 40%. Alguns passos foram realizados a fim de melhorar a precisão
da rede, como adição de mais dados na base de treinamento, técnicas de data augumentation, adição de épocas e camadas
LSTM. Porém, mesmo após todas essas tentativas, a rede continuou instável, errando até mesmo as predições.<br>

Com esta barreira encontrada, a solução alternativa desenhada para a entrega do projeto em tempo hábil, foi a criação
de dois modelos. Um que servirá de exemplo para as melhorias futuras, contemplando a identificação dos gestos em Libras
e outro modelo baseado neste primeiro, porém simplificado, que contempla apenas sinais estáticos da linguagem. A
metodologia aplicado em ambos será detalhada logo abaixo.

### 1. Coleta de dados
Para as primeiras implementações do projeto, o dataset foi montado com imagens em formato png. Foram tiradas 500 fotos
de cada sinal do alfabeto em Libras através da ferramenta [Teachable Machine](https://teachablemachine.withgoogle.com/).
As imagens foram divididas em uma proporção 75:25 em duas pastas denominadas <code>train</code> e <code>test</code>.<br>

Já para a implementação do projeto utilizando LSTM, as ferramentas do mediapipe e do openCV foram utilizadas. No caso do projeto
que mapeia os gestos, foram coletados os pontos de referência das duas mãos, da pose e da face como mostra na imagem 
abaixo. Para o modelo que mapeia as imagens estáticas, foram coletados apenas os pontos da mão direita.

![](https://github.com/DayaneCordeiro/libras-recognition-system/blob/main/imgs/landmarks_stylized.png)

Para o projeto com os gestos, foi implementada uma lógica que cria uma pasta para cada classe definida e 30 
pastas para cada uma das classes, onde ficam as sequências de frames coletadas. A ideia geral foi, coletar 30 frames 
dos gestos executados para assim formar uma espécie de vídeo, que por sua vez, foi salvo em formato npy. Sendo assim, 
a estrutura da base de dados ficou parecida o que se vê abaixo:

> dataset/<br>
>> a/ <br>
>>> 0/
>>>> trinta_frames_da_classe_a.npy<br>

>>> 1/<br>
>>>> outros_trinta_frames_da_classe_a.npy<br>

Para o modelo LSTM que implementa apenas as imagens estáticas, foram coletadas 400 imagens de cada um dos sinais. 
A base de dados, portanto foi simplificada e ficou da seguinte forma:

> dataset/<br>
>> a/ <br>
>>>> 0.npy<br>
>>>> 1.npy

### 2. Divisão da base de dados

Para ambas redes LSTM, os dados foram divididos entre 80% para treinamento e 20% para teste e validação da rede.

### 3. Arquitetura da rede

Como citado anteriormente, várias arquiteturas foram testadas. No caso dos gestos, a arquitetura será um ponto de
melhorias, já para a rede que trata apenas dados estáticos, a definição da rede foi altamente satisfatória. A
definição da arquitetura ficou da seguinte forma:

```
INPUT => LSTM => LSTM => LSTM => DENSE => DENSE => DENSE => OUTPUT
```

Para realizar o treinamento, foram configuradas inicialmente 2000 épocas, porém para mitigar problemas de overfitting,
foi adicionada uma lógica que para o treinamento da rede assim que seja atingido um valor pré-definido de acurácia, que
no caso deste projeto, foi de 90%.

### 4. Predições e logs

Após treinamento da rede neural, foi feita uma predição com algum valor aleatório passando pela rede. Além disso,
foram coletados logs que mostram acurácia em relação à quantidade de épocas utilizadas.

### 5. Testes em tempo real
A última etapa, trata-se do uso da aplicação em si. Novamente obtendo apoio do mediapipe e do openCV para coletar as
imagens da webcam e mapear os pontos de interesse. Com os pontos definidos, a imagem é classificada e a letra ou
gesto detectado é exibido na tela.

## Resultados obtidos

A seguir, são exibidos os resultados obtidos para os dois modelos LSTM.

### Modelo estático:
A rede neural foi implementada com um valor inicial de 2000 épocas, porém, foi utilizada uma lógica de parada adiantada 
(Early Stopping Callback), caso alcançasse uma acurácia de no mínimo 90%. Após 35 épocas executadas, o valor desejado 
para a acuária foi atingido, a imagem abaixo mostra os logs retornados ao fim do treinamento da rede.

![](https://github.com/DayaneCordeiro/libras-recognition-system/blob/main/imgs/resultados_treino_rede.png)

<br>
Gráfico coletado pós treinamento da rede, onde no eixo x é exibida a quantidade de épocas e no eixo y a acurácia:

![](https://github.com/DayaneCordeiro/libras-recognition-system/blob/main/imgs/gr%C3%A1fico_acuracia_por_epoca_modelo_estatico.png)

Grafo gerado pelo treinamento da rede:

![](https://github.com/DayaneCordeiro/libras-recognition-system/blob/main/imgs/png.png)

### Modelo dinâmico:
Foi implemtada com a mesma lógica de parada antecipada da rede estática. Para o treinamento da rede com 3 classes com acurácia acima de 90%, foram necessárias 81 épocas.

![](https://github.com/DayaneCordeiro/libras-recognition-system/blob/main/imgs/resultado_rede_dinamica.png)

<br>
Gráfico coletado pós treinamento da rede, onde no eixo x é exibida a quantidade de épocas e no eixo y a acurácia:

![](https://github.com/DayaneCordeiro/libras-recognition-system/blob/main/imgs/grafico_rede_dinamica.png)

Grafo gerado pelo treinamento da rede:

![](https://github.com/DayaneCordeiro/libras-recognition-system/blob/main/imgs/png%20(1).png)

## Licença

Este projeto está licenciado sob a MIT License. Veja o arquivo LICENSE para mais detalhes.

---
<div id="author">
    <h1>Autora</h1>
    <a href="https://github.com/DayaneCordeiro">
        <img style="border-radius: 50%;" src="https://avatars.githubusercontent.com/u/50596100?v=4" width="150px;" alt=""/>
        <br />
        <sub><b>Dayane Cordeiro</b></sub>
    </a>
</div>

Made with ❤️ by Dayane Cordeiro!

✔ Computer Engineering student at PUC Minas<br>
✔ Java Developer<br>
✔ Passionate about software development, computer architecture and learning.<br>
