# Rede de adsorção — pacote de inferência e treino

Este diretório reúne o pacote da rede em cascata: contrato de entrada, atributos derivados, modelos, exemplo de inferência e código para regenerar os dados e treinar as redes de tempos e de forma.

## Estrutura

- `codigo/contrato/`: contrato dos dados, solver, geração e preparação do conjunto de treino.
- `codigo/treino/`: `train_times.py` e `train_forma.py`.
- `modelos/`: pesos e normalizadores usados pela inferência; `holdout_10k.npz` serve para uma verificação rápida do pipeline.
- `dados/`: destino do cache de treino gerado localmente.
- `saidas/`: modelos, previsões e outros resultados criados durante o treino. Esta pasta é ignorada pelo Git.

A rede usa os 31 parâmetros físicos e 16 atributos derivados (47 entradas no total). A primeira rede prevê três grandezas de tempo/progressão; a segunda prevê 100 pontos temporais para cada um de dois perfis (200 saídas). O solver de referência usa 71 nós axiais.

## Ambiente

O ambiente de referência do pacote é Python 3.10. Instale as dependências dentro deste diretório:

```bash
python -m venv .venv
# Ative o ambiente virtual e então:
python -m pip install -r requirements.txt
```

A inferência de exemplo pode ser executada com:

```bash
python codigo/example_usage.py
```

## Treino rápido para conferir se os scripts funcionam

O arquivo `modelos/holdout_10k.npz` já contém os campos necessários para exercitar os dois scripts sem gerar o conjunto de 1,1 milhão de casos. O exemplo abaixo usa uma época e grava tudo em `saidas/`. **É apenas um teste técnico do pipeline; esse arquivo é um conjunto de holdout e o resultado não deve ser usado para medir desempenho nem como modelo final.**

No PowerShell, a partir de `network_2x1/`:

```powershell
$env:CACHE = "modelos/holdout_10k.npz"
$env:EPOCHS = "1"
$env:CKPT_EVERY = "0"
$env:MODELO_OUT = "saidas/smoke_tempos.keras"
$env:PRED_OUT = "saidas/smoke_tempos_preditos.npz"
python codigo/treino/train_times.py

$env:PRED = "saidas/smoke_tempos_preditos.npz"
$env:MODELO_OUT = "saidas/smoke_forma.keras"
python codigo/treino/train_forma.py
```

No Linux/WSL, os mesmos valores podem ser passados na linha de comando:

```bash
CACHE=modelos/holdout_10k.npz EPOCHS=1 CKPT_EVERY=0 MODELO_OUT=saidas/smoke_tempos.keras PRED_OUT=saidas/smoke_tempos_preditos.npz python -u codigo/treino/train_times.py
CACHE=modelos/holdout_10k.npz EPOCHS=1 CKPT_EVERY=0 PRED=saidas/smoke_tempos_preditos.npz MODELO_OUT=saidas/smoke_forma.keras python -u codigo/treino/train_forma.py
```

A conclusão dos dois comandos confirma que os scripts carregaram os dados, montaram as redes, treinaram e salvaram os artefatos de teste. As métricas de uma época não são uma avaliação científica.

## Treino com o conjunto completo

O pacote não inclui o banco bruto nem `dados/cache.npz`: o banco completo ocuparia centenas de GB. O README do pacote original descreve o conjunto reprodutível por sementes. Para regenerá-lo, use `codigo/contrato/generate_1M_blocks.sh` em Linux/WSL com o ambiente JAX configurado, depois execute o funil de máscara, pré-processamento, cálculo dos tempos, filtragem e concatenação (`valid_mask.py`, `preprocess_filtered.py`, `level.py`, `screen.py` e `concat.py`). O cache final esperado é `dados/cache.npz`.

Com o cache completo pronto, execute a partir deste diretório:

```bash
CACHE=dados/cache.npz python -u codigo/treino/train_times.py
CACHE=dados/cache.npz PRED=saidas/tempos_preditos.npz python -u codigo/treino/train_forma.py
```

Os scripts aceitam variáveis de ambiente para ajustar `EPOCHS`, `BATCH`, caminhos de entrada/saída e demais parâmetros. A configuração padrão treina as duas redes completas. Os arquivos brutos, cache, checkpoints e resultados permanecem locais e não são versionados.
