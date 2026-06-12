# BRAYAN

Asistente de IA multimodal (voz + visión) en el dispositivo. Mantén conversaciones naturales con voz y video con una IA que corre completamente de forma local en tu máquina.

Este proyecto está basado en el repositorio original [Parlor de fikrikarim](https://github.com/fikrikarim/parlor), sirviendo como la base para nuestro desarrollo y optimizaciones.

BRAYAN utiliza [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) para el procesamiento y comprensión multimodal (visión y lenguaje) y [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) para la síntesis de voz (Text-to-Speech). Tú hablas, le muestras tu cámara y él te responde hablando, todo de forma 100% local y privada.

---

## Características Principales y Mejoras

- **Detección de Actividad de Voz (VAD) en el navegador:** Utiliza [Silero VAD](https://github.com/ricky0123/vad) en el cliente. Manos libres, sin necesidad de presionar un botón para hablar.
- **Bypass de Webcam Virtual:** Implementa un puente WebRTC dinámico que extrae el `getUserMedia` nativo de Chrome para evitar que extensiones de webcam virtual (como *LookGood Live*) bloqueen o eliminen el canal de audio del micrófono.
- **Arquitectura de Sincronización en GPU:** Procesamiento seguro a través de un ejecutor de hilo único (`ThreadPoolExecutor(max_workers=1)`) en el backend que protege el ciclo de vida de la GPU (Metal/OpenCL) ante recargas rápidas, eliminando los bloqueos (*deadlocks*) permanentes de GPU.
- **Interrupción por Voz (Barge-in):** Puedes interrumpir a la IA en medio de su respuesta con solo hablarle.
- **Streaming de audio por oraciones:** El audio comienza a reproducirse progresivamente antes de que termine de generarse la respuesta completa.

---

## Requerimientos del Dispositivo

- **Sistema Operativo:** macOS (Apple Silicon M1, M2, M3, M4) o Linux con GPU compatible.
- **Memoria RAM:** Al menos 3 GB de RAM libre dedicada para el modelo.
- **Procesador (CPU):** Optimizado para chips de la serie M de Apple (utiliza aceleración Metal GPU de forma nativa) o CPU multinúcleo en Linux.
- **Almacenamiento:** Aproximadamente 3 GB de espacio libre (los modelos se descargan automáticamente en la primera ejecución: ~2.58 GB para Gemma 4 E2B, más los pesos de TTS de Kokoro).
- **Python:** Versión 3.12 o superior.

---

## Instrucciones de Instalación y Uso rápido

### 1. Clonar el repositorio
```bash
git clone https://github.com/cfarias73/parlor2.git
cd parlor2
```

### 2. Instalar el gestor de paquetes `uv` (Recomendado)
`uv` es un instalador y gestor de paquetes de Python ultrarrápido escrito en Rust.
- **En macOS/Linux:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### 3. Sincronizar dependencias e inicializar
Entra al directorio `src/` e instala el entorno virtual con todas las dependencias requeridas:
```bash
cd src
uv sync
```

### 4. Lanzar el servidor
Ejecuta el servidor de desarrollo en puerto local:
```bash
uv run server.py
```
*El servidor leerá la configuración del puerto desde el archivo `.env` (por defecto `3436`).*

### 5. Abrir en el navegador
Abre tu navegador (Google Chrome recomendado) en:
[http://localhost:3436](http://localhost:3436)

Concede permisos de acceso a la cámara y al micrófono cuando el sistema y el navegador lo soliciten. ¡Y listo! Comienza a hablarle en español o inglés (puedes alternar el idioma en el selector inferior).

---

## Lanzador de Escritorio (macOS)

El repositorio incluye un script lanzador de acceso rápido `Parlor.command`. 
Para usarlo:
1. Asegúrate de mover la carpeta del proyecto a la ubicación deseada.
2. Abre el archivo `Parlor.command` con un editor de texto y actualiza la ruta en la línea 3 para que apunte al directorio `src/` de tu nueva ubicación.
3. Haz doble clic en el script para iniciar el servidor y abrir el navegador web automáticamente en un solo paso.

---

## Configuración y Variables de Entorno

Puedes configurar las variables creando un archivo `.env` en la carpeta raíz o dentro de `src/`:

| Variable | Por defecto | Descripción |
| :--- | :--- | :--- |
| `PORT` | `3436` | Puerto de escucha del servidor FastAPI. |
| `MODEL_PATH` | Descarga automática | Ruta local alternativa hacia el archivo del modelo `gemma-4-E2B-it.litertlm`. |

---

## Rendimiento de Referencia (Apple M3 Pro)

| Fase de Procesamiento | Tiempo estimado |
| :--- | :--- |
| Comprensión de voz y visión | ~1.8 - 2.2s |
| Generación de respuesta (~25 tokens) | ~0.3s |
| Síntesis de voz a texto (TTS) | ~0.3 - 0.7s |
| **Latencia Total End-to-End** | **~2.5 - 3.0s** |

*Velocidad de descodificación de texto: ~83 tokens/segundo en GPU de M3 Pro.*

---

## Estructura del Proyecto

```
parlor2/
├── src/
│   ├── server.py        # Servidor FastAPI WebSocket e inferencia de Gemma 4
│   ├── tts.py           # Conexión de TTS (MLX en Mac, ONNX en Linux)
│   ├── index.html       # Interfaz de usuario (VAD, webcam, reproductor de audio)
│   ├── pyproject.toml   # Definición de dependencias
│   └── benchmarks/      # Scripts de pruebas y evaluación de rendimiento
└── Parlor.command       # Lanzador rápido para macOS
```

---

## Agradecimientos

- Repo base: [Parlor de fikrikarim](https://github.com/fikrikarim/parlor)
- [Gemma 4](https://ai.google.dev/gemma) desarrollado por Google DeepMind.
- [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) por Google AI Edge.
- [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) TTS por Hexgrad.
- [Silero VAD](https://github.com/snakers4/silero-vad) para la detección de voz en el navegador.

---

## Licencia

Este proyecto está licenciado bajo la Licencia **Apache 2.0**. Consulta el archivo `LICENSE` para más detalles.
