$projectDir = 'C:\Ankshit\CVC-MAGNet-V2'
$pythonExe = 'C:\Users\M S I\anaconda3\python.exe'
$reviewUrl = 'http://127.0.0.1:8766/'
try {
    $response = Invoke-RestMethod ($reviewUrl + 'api/manifest') -TimeoutSec 2
    if ($response.mask_type -ne 'Model-predicted CVC segmentation; no annotation overlays') { throw 'Unexpected service on review port' }
} catch {
    Start-Process -FilePath $pythonExe -ArgumentList @('-m','uvicorn','tools.mask_review.server:app','--host','127.0.0.1','--port','8766') -WorkingDirectory $projectDir -WindowStyle Hidden -RedirectStandardOutput "$projectDir\outputs\predicted_mask_review_expanded\server.log" -RedirectStandardError "$projectDir\outputs\predicted_mask_review_expanded\server_error.log"
    Start-Sleep -Seconds 3
}
Start-Process $reviewUrl
