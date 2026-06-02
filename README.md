# Processador de DXF para Maquetes

Aplicacao Streamlit para transformar ficheiros DXF em camadas de maquete, preparar folhas de corte laser e exportar resultados em DXF/SVG.

O projeto foi pensado para maquetes topograficas ou volumetricas feitas por empilhamento de placas. A aplicacao le polilinhas fechadas e pontos 3D do DXF, gera camadas, permite escolher entre corte solido e corte vazado, calcula nesting em folhas de material e gera ficheiros prontos para maquina laser.

## About This Site

Ferramenta Streamlit para transformar ficheiros DXF em camadas de maquete, preparar folhas de corte laser e exportar resultados em DXF/SVG.

Inclui modos solido e vazado, margem de base/cola, linhas de gravacao para montagem, preview 2D/3D, nesting automatico e geracao opcional de base ate a cota real.

Projeto em desenvolvimento ativo. Antes de cortar, confirme sempre os ficheiros exportados num software CAD.

Licenca: GNU General Public License v3.0 ou posterior (GPL-3.0-or-later).

## Funcionalidades

- Importacao de polilinhas fechadas DXF como geometrias de corte.
- Leitura de pontos 3D para atribuicao de cotas.
- Geracao de camadas por curvas de nivel/topografia.
- Geracao opcional de camadas de base ate a cota real da maquete.
- Dois modos de maquete:
  - **Opcao A - Solido**: empilhamento cheio.
  - **Opcao B - Aneis/Vazado**: gera areas visiveis e areas de base/cola.
- Margem de colagem para a Opcao B.
- Linhas de gravacao para apoio de montagem.
- Preview das camadas geradas em 2D.
- Preview 3D das camadas empilhadas.
- Nesting automatico em folhas com margem de seguranca.
- Exportacao de folhas individuais em DXF e SVG.
- Exportacao de DXF unico com todas as folhas.
- Diagnostico do DXF exportado para validar layers, entidades e sobreposicoes.
- Simplificacao de linhas para reduzir pontos desnecessarios no corte laser.

## Requisitos

- Python 3.11 recomendado.
- Windows, macOS ou Linux.
- Ficheiro DXF com polilinhas fechadas para as formas.
- Opcionalmente, pontos 3D no DXF para atribuir cotas automaticamente.

Dependencias principais:

- Streamlit
- ezdxf
- Shapely
- Plotly
- rectpack
- NetworkX

## Instalacao

Clone o repositorio e entre na pasta do projeto:

```bash
git clone <url-do-repositorio>
cd <nome-do-repositorio>
```

Crie e ative um ambiente virtual:

```bash
python -m venv venv
```

No Windows:

```bash
venv\Scripts\activate
```

No macOS/Linux:

```bash
source venv/bin/activate
```

Instale as dependencias:

```bash
pip install -r requirements.txt
```

## App Online

A aplicacao esta disponivel em:

https://topomaquette.streamlit.app/

## Como Executar

Execute a aplicacao com Streamlit:

```bash
streamlit run main.py
```

Depois abra o endereco indicado no terminal, normalmente:

```text
http://localhost:8501
```

## Ficheiros de Exemplo

A pasta `examples/` inclui um DXF simples gerado pelo helper de testes e um DXF complexo para testar a aplicacao em condicoes mais proximas de um caso real:

- `examples/test_topography_generated.dxf`: exemplo simples gerado por `generate_test_dxf.py`, com boundary igual a camada mais baixa, poligonos fechados e pontos 3D em cotas 0.0, 0.5, 1.0, 1.5 e 2.0 para validar rapidamente importacao, cotas, topologia e nesting basico.
- `examples/topo_c_teste_complexo.dxf`: exemplo maior para testar desempenho, muitas camadas e nesting em caso realista.

Mais detalhes estao em `examples/README.md`.

## Fluxo de Uso

1. Importe o ficheiro DXF.
2. Escolha as layers relevantes: boundary, curvas/formas e pontos 3D quando existirem.
3. Ajuste escala, cotas e parametros da maquete.
4. Escolha o tipo de corte:
   - **Solido (Opcao A)** para empilhamento cheio.
   - **Aneis (Opcao B)** para pecas vazadas com zona de base/cola.
5. Clique em **Gerar Formas**.
6. Verifique o preview das camadas geradas.
7. Configure cama, margem, folgas, nesting e simplificacao de linhas.
8. Clique em **Gerar Folhas de Corte**.
9. Verifique as folhas geradas e exporte DXF/SVG.
10. Use o diagnostico DXF antes de enviar para corte laser.

## Estrutura do Projeto

```text
.
|-- main.py                 # Interface Streamlit e fluxo principal da aplicacao
|-- requirements.txt        # Dependencias Python
|-- test_core.py            # Testes unitarios do nucleo geometrico/exportacao
|-- generate_test_dxf.py    # Gera o DXF local usado por alguns testes
`-- core/
    |-- parser.py           # Leitura de poligonos e pontos 3D em DXF
    |-- topology.py         # Relacoes topologicas entre formas
    |-- slicer.py           # Geracao de aneis/camadas por topologia
    |-- terrain_slicer.py   # Geracao de camadas topograficas por cotas
    |-- nesting.py          # Divisao e arrumacao das pecas em folhas
    |-- exporter.py         # Exportacao DXF/SVG e diagnostico de ficheiros
    |-- diagnostics.py      # Diagnosticos de geometrias importadas
    |-- elevation_points.py # Estimativa de cotas por pontos
    `-- ui_utils.py         # Funcoes auxiliares da interface
```

## Testes

Para correr os testes:

```bash
python -m unittest test_core.py
```

O ficheiro `test_core.py` valida o nucleo geometrico, nesting, exportacao DXF/SVG e regras das Opcoes A/B. O auxiliar `generate_test_dxf.py` e mantido no repositorio porque permite recriar o ficheiro local `test_topography.dxf` usado pelos testes quando ele nao existe, e tambem gerar o exemplo simples `examples/test_topography_generated.dxf`.

Scripts soltos de debug ou investigacao local, como `debug.py`, `plot_debug.py` e `analyze_topo.py`, ficam ignorados pelo Git e nao fazem parte do deploy.

## Notas para DXF

- As formas devem ser polilinhas fechadas.
- A layer de boundary deve conter o limite exterior da maquete/material de referencia.
- Os pontos 3D devem estar em entidades `POINT` com coordenada Z preenchida.
- Para corte laser, confirme sempre o diagnostico e abra o DXF/SVG num software CAD antes de produzir.

## Estado do Projeto

Ferramenta em desenvolvimento ativo para preparacao de maquetes por corte laser. Algumas operacoes geometricas dependem da qualidade do DXF original, principalmente polilinhas abertas, geometrias invalidas, excesso de pontos e curvas muito proximas.

## Licenca

Este projeto e distribuido sob a GNU General Public License v3.0 ou posterior (GPL-3.0-or-later). Ver [LICENSE](LICENSE).
