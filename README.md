# Identificação de Sinais e Gestos de Libras com Redes Neurais LSTM

## Visão Geral

Este projeto visa identificar sinais e gestos da Linguagem de Sinais Brasileira (Libras) utilizando redes neurais LSTM (Long Short-Term Memory) e a ferramenta MediaPipe. Através deste projeto, esperamos contribuir para a acessibilidade e inclusão de pessoas com deficiência auditiva, facilitando a comunicação entre falantes de Libras e ouvintes.

## Linguagem de Sinais Brasileira (Libras)

A Libras é a língua de sinais utilizada pela comunidade surda no Brasil. Assim como qualquer língua, ela possui sua própria gramática e sintaxe, sendo composta por sinais feitos com as mãos, expressões faciais e movimentos corporais. A Libras é reconhecida oficialmente como meio legal de comunicação e expressão no Brasil, tendo sido incluída na Lei de Diretrizes e Bases da Educação Nacional.

## LSTM (Long Short-Term Memory)

LSTM é um tipo de rede neural recorrente (RNN) projetada para aprender dependências de longo prazo. Diferente das RNNs tradicionais, as LSTMs são capazes de armazenar informações por longos períodos de tempo, graças a uma arquitetura especial que inclui células de memória e mecanismos de controle como gates de entrada, saída e esquecimento. Isso as torna particularmente úteis para tarefas de série temporal e sequências, como reconhecimento de fala, tradução de idiomas e, neste caso, identificação de sinais de linguagem de sinais.

## Ferramentas e Tecnologias

- **Python**: A linguagem de programação utilizada para desenvolver o projeto.
- **MediaPipe**: Um framework de machine learning para construir pipelines multimodais customizáveis. Utilizado aqui para captura e processamento dos movimentos das mãos.
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

## Licença

Este projeto está licenciado sob a MIT License. Veja o arquivo LICENSE para mais detalhes.

## Contato

Para mais informações, entre em contato pelo email: [dayane.cordeirogs@gmail.com](mailto:dayane.cordeirogs@gmail.com).

---

Esperamos que este projeto ajude a promover a inclusão e facilite a comunicação entre falantes de Libras e ouvintes. Agradecemos por seu interesse e apoio!
