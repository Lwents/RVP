# Docker Setup Guide

Huong dan nay dung cho Windows may khac de cai Docker Desktop va chay du an Auto-Translate AI.

## 1. Yeu cau

- Windows 10/11 64-bit.
- Quyen Administrator de cai Docker Desktop.
- Nen bat WSL2/Virtualization trong BIOS/Windows.
- Neu muon dung GPU NVIDIA trong Docker, can cai driver NVIDIA moi va Docker Desktop ho tro WSL2 GPU.

## 2. Cai Docker Desktop

Mo PowerShell va chay:

```powershell
winget install --id Docker.DockerDesktop -e --accept-package-agreements --accept-source-agreements
```

Neu hien popup UAC, bam `Yes`.

Sau khi cai xong:

1. Mo Docker Desktop.
2. Doi den khi Docker bao engine dang chay.
3. Neu Docker yeu cau logout/restart Windows, hay lam theo.
4. Dong va mo lai PowerShell/VS Code terminal.

Kiem tra:

```powershell
docker --version
docker compose version
```

Neu len version la Docker da san sang.

## 3. Chay du an bang Docker

Vao thu muc du an:

```powershell
cd C:\Users\kirit\Documents\AI_Video
```

Build image lan dau:

```powershell
.\build-docker.ps1
```

Chay du an:

```powershell
.\start-docker.ps1
```

Mo trinh duyet:

- Frontend: http://127.0.0.1:5173
- Backend docs: http://127.0.0.1:8000/docs

## 4. Cac lan sau

Neu da build roi, chi can:

```powershell
.\start-docker.ps1
```

Dung container:

```powershell
docker compose down
```

Xem trang thai:

```powershell
docker compose ps
```

Xem log:

```powershell
docker compose logs -f
```

## 5. Neu GPU Docker loi

Mac dinh `docker-compose.yml` dung GPU NVIDIA:

```powershell
docker compose up -d
```

Neu Docker bao loi GPU/NVIDIA, chay ban CPU:

```powershell
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

Ban CPU cham hon nhung de chay on dinh hon tren may chua cau hinh GPU Docker.

## 6. Du lieu render nam o dau?

Thu muc nay duoc mount vao container:

```text
backend/storage
```

Tat ca job, video tai ve, phu de, audio, output render se nam trong do.

## 7. Loi thuong gap

### `docker is not recognized`

Dong terminal cu, mo terminal moi. Neu van loi, kiem tra Docker Desktop da cai va da mo chua.

### Docker Desktop bi dung o trang starting

Thu restart Windows. Neu van loi, mo PowerShell admin va chay:

```powershell
wsl --update
wsl --shutdown
```

Sau do mo lai Docker Desktop.

### Port da duoc dung

Neu 8000 hoac 5173 dang bi app khac chiem:

```powershell
docker compose down
```

Hoac sua port trong `docker-compose.yml`.

