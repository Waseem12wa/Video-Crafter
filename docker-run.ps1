# Script to run VideoCrafter Docker container with proper resource limits
# This prevents server crashes due to memory exhaustion

Write-Host "=================================" -ForegroundColor Cyan
Write-Host "  VideoCrafter Docker Launcher  " -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Stop and remove old container
Write-Host "[1/4] Stopping old container..." -ForegroundColor Yellow
docker rm -f videocrafter 2>$null
Write-Host "     Done!" -ForegroundColor Green
Write-Host ""

# Pull latest image
Write-Host "[2/4] Pulling latest image from Docker Hub..." -ForegroundColor Yellow
docker pull waseemzahid48/videocrafter-app:latest
Write-Host "     Done!" -ForegroundColor Green
Write-Host ""

# Run container with resource limits
Write-Host "[3/4] Starting container with resource limits..." -ForegroundColor Yellow
docker run -d `
  -p 5001:5000 `
  --name videocrafter `
  --memory="4g" `
  --memory-swap="6g" `
  --cpus="2.0" `
  --restart unless-stopped `
  -v videocrafter-uploads:/app/uploads `
  -v videocrafter-outputs:/app/outputs `
  -v videocrafter-clips:/app/clips `
  waseemzahid48/videocrafter-app:latest

if ($LASTEXITCODE -eq 0) {
    Write-Host "     Done!" -ForegroundColor Green
    Write-Host ""
    
    # Show status
    Write-Host "[4/4] Container status:" -ForegroundColor Yellow
    Start-Sleep -Seconds 2
    docker ps --filter "name=videocrafter"
    Write-Host ""
    
    Write-Host "=================================" -ForegroundColor Cyan
    Write-Host "Server is running at:" -ForegroundColor Green
    Write-Host "  http://localhost:5001" -ForegroundColor White
    Write-Host ""
    Write-Host "To view logs:" -ForegroundColor Yellow
    Write-Host "  docker logs -f videocrafter" -ForegroundColor White
    Write-Host ""
    Write-Host "To stop:" -ForegroundColor Yellow
    Write-Host "  docker stop videocrafter" -ForegroundColor White
    Write-Host "=================================" -ForegroundColor Cyan
} else {
    Write-Host "     Failed to start container!" -ForegroundColor Red
    Write-Host "     Check Docker logs for details" -ForegroundColor Red
}
