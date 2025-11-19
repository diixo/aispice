# ASPICE assessment, compliance, excellence

|  Command                        | Description                      |
| --------------------------------|----------------------------------|
| docker compose up -d            | Запускает, если всё уже собрано
| docker compose up -d --build    | Пересобирает и запускает
| docker compose build            | Только собирает, не запускает
| docker compose down             | Останавливает и удаляет контейнеры
| docker exec -it aspice_app bash | Зайти внутрь контейнера


Requirements:
```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install triton-windows
pip install xgramm
```
