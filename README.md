# Optimización de Horarios Académicos con Algoritmos Genéticos

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 📝 Descripción
Implementación de un algoritmo genético en Python para resolver el problema de planificación de horarios académicos, optimizando la distribución de asignaturas cumpliendo restricciones duras y preferencias blandas. Desarrollado como proyecto para la asignatura de Inteligencia Artificial.

## 🎯 Problema a Resolver
Dado:
- `N` asignaturas con `h_i` horas semanales cada una
- `M` días disponibles con `k` horas por día

**Restricciones duras**:
1. ❌ No solapamiento de asignaturas en misma hora/día
2. ⏱ Máximo 2 horas/día por asignatura

**Objetivos de optimización**:
1. 🕳 Minimizar huecos vacíos entre clases
2. 📅 Reducir días utilizados (horario compacto)
3. 🔄 Agrupar horas consecutivas de misma asignatura
