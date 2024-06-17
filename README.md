# Identificação de Sinais e Gestos de Libras com Redes Neurais LSTM

## Visão Geral

Este projeto visa identificar sinais e gestos da Linguagem de Sinais Brasileira (Libras) utilizando redes neurais LSTM (Long Short-Term Memory) e a ferramenta MediaPipe. Através deste projeto, esperamos contribuir para a acessibilidade e inclusão de pessoas com deficiência auditiva, facilitando a comunicação entre falantes de Libras e ouvintes.

## Linguagem de Sinais Brasileira (Libras)

A Libras é a língua de sinais utilizada pela comunidade surda no Brasil. Assim como qualquer língua, ela possui sua própria gramática e sintaxe, sendo composta por sinais feitos com as mãos, expressões faciais e movimentos corporais. A Libras é reconhecida oficialmente como meio legal de comunicação e expressão no Brasil, tendo sido incluída na Lei de Diretrizes e Bases da Educação Nacional.

## LSTM (Long Short-Term Memory)

LSTM é um tipo de rede neural recorrente (RNN) projetada para aprender dependências de longo prazo. Diferente das RNNs tradicionais, as LSTMs são capazes de armazenar informações por longos períodos de tempo, graças a uma arquitetura especial que inclui células de memória e mecanismos de controle como gates de entrada, saída e esquecimento. Isso as torna particularmente úteis para tarefas de série temporal e sequências, como reconhecimento de fala, tradução de idiomas e, neste caso, identificação de sinais de linguagem de sinais.

## Ferramentas e Tecnologias

- **Python**: A linguagem de programação utilizada para desenvolver o projeto - versão 3.8.
- **MediaPipe**: Um framework de machine learning para construir pipelines multimodais customizáveis. Utilizado aqui para captura e processamento dos movimentos das mãos, pose e rosto.
- **TensorFlow/Keras**: Utilizados para a construção e treinamento da rede neural LSTM.

## Estrutura do Projeto

1. **Captura de Dados**: Utilizando a ferramenta MediaPipe, capturamos os movimentos das mãos em tempo real, que são então convertidos em coordenadas espaciais.
2. **Processamento de Dados**: As coordenadas capturadas são processadas e formatadas para serem utilizadas no treinamento do modelo LSTM.
3. **Treinamento do Modelo**: A rede neural LSTM é treinada utilizando um conjunto de dados de sinais de Libras, permitindo que ela aprenda a reconhecer e classificar os gestos corretamente.
4. **Inferência e Reconhecimento**: O modelo treinado é utilizado para identificar sinais em tempo real, traduzindo gestos de Libras para texto ou fala.

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

## Resultados obtivos
### Modelo estático:
A rede neural foi implementada com um valor incial de 2000 épocas, porém, foi utilizada uma lógica de parada adiantada (Early Stopping Callback), caso alcançasse uma acurácia de no mínimo 90%. Após 35 épocas executadas, o valor desejado para a acuária foi atingido, a imagem abaixo mostra os logs retornados ao fim do treinamento da rede.
![](https://github.com/DayaneCordeiro/libras-recognition-system/blob/main/imgs/resultados_treino_rede.png)

<br>

Para uma visualização gráfica dos logs, foi utilizada a classe TensorBoard da biblioteca <code>keras.src.callbacks</code>. Para execução da biblioteca é necessário seguir os seguintes passos:
1. Na pasta raiz do projeto, navegue ate o diretório de logs:
   ```bash
   cd Logs/train
   ```
2. Execute o seguinte comando:
   ```bash
   tensorboard --logdir=.
   ```
3. Após a execução do comando, abra o navegador no sequinte endereço:
   ```bash
   http://localhost:6006
   ```

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

Made with ❤️ by Dayane Cordeiro!

✔ Computer Engineering student at PUC Minas<br>
✔ Java Developer<br>
✔ Passionate about software development, computer architecture and learning.<br>
