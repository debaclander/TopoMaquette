# Exemplos

Esta pasta contem um DXF simples gerado pelos testes e um DXF de referencia para testar a aplicacao em condicoes mais proximas de um caso real.

## Ficheiros

- `test_topography_generated.dxf`
  - Gerado por `generate_test_dxf.py`.
  - 1 boundary igual a camada mais baixa.
  - 4 poligonos fechados acima da base.
  - 13 pontos 3D para atribuicao automatica de cotas.
  - Cotas de teste: 0.0, 0.5, 1.0, 1.5 e 2.0.
  - Layers principais: `boundary`, `topografia`, `3D_PONTOS`.
  - Bom para validar rapidamente importacao, pontos 3D, topologia, Opcoes A/B, preview e nesting basico.

- `topo_c_teste_complexo.dxf`
  - Exemplo maior baseado no ficheiro `TOPO_C.dxf`.
  - 48 poligonos fechados.
  - 498 pontos 3D.
  - Layers principais: `BOUNDARY`, `TOPOGRAPHY`, `3D_PONTOS`.
  - Bom para testar desempenho, geracao de muitas camadas, Opcao B e nesting em caso realista.

## Layers Usadas

No exemplo simples:

- `boundary`
- `topografia`
- `3D_PONTOS`

No exemplo complexo:

- `BOUNDARY`
- `TOPOGRAPHY`
- `3D_PONTOS`

## Como Testar

1. Execute a aplicacao:

```bash
streamlit run main.py
```

2. Importe um dos ficheiros DXF desta pasta.
3. Para `test_topography_generated.dxf`, selecione:
   - boundary: `boundary`
   - curvas: `topografia`
   - pontos 3D: `3D_PONTOS`
4. Para `topo_c_teste_complexo.dxf`, selecione:
   - boundary: `BOUNDARY`
   - curvas: `TOPOGRAPHY`
   - pontos 3D: `3D_PONTOS`
5. Clique em `Atribuir cotas pelos pontos 3D`.
6. Clique em `Gerar Formas`.
7. Gere as folhas de corte para testar nesting/exportacao.
