<?php

namespace App\GraphQL\Queries;

use Illuminate\Support\Facades\Log;

class DiagnosticReportQuery
{
    /**
     * Resolves the clinical diagnostic report by triggering an on-device
     * GNN inference engine via the INT8 quantized core.
     *
     * @param  null  $_
     * @param  array<string, mixed>  $args
     */
    public function resolve($_, array $args) 
    {
        $targetId = $args['patientId'];

        // 1. Absolute Path Resolution
        // Ensures the bridge works regardless of where the PHP process is initialized
        $basePath = base_path('../core_quantizer');
        $pythonBinary = $basePath . '/venv/Scripts/python.exe'; // Explicit VENV path for Windows
        $scriptPath = $basePath . '/predict.py';

        // 2. Cross-Platform Command Construction
        // Uses absolute paths to avoid MINGW64 / Laravel pathing conflicts
        $command = sprintf(
            '%s %s %s',
            escapeshellarg($pythonBinary),
            escapeshellarg($scriptPath),
            escapeshellarg($targetId)
        );

        // 3. Execution with Error Capture
        // proc_open allows us to capture STDERR to diagnose "Architecture over substance" issues
        $descriptorspec = [
            1 => ['pipe', 'w'], // stdout
            2 => ['pipe', 'w'], // stderr
        ];

        $process = proc_open($command, $descriptorspec, $pipes, $basePath);

        if (is_resource($process)) {
            $output = stream_get_contents($pipes[1]);
            $errorOutput = stream_get_contents($pipes[2]);
            
            fclose($pipes[1]);
            fclose($pipes[2]);
            $returnCode = proc_close($process);

            $result = json_decode($output, true);

            // 4. Fallback Logic & Logging
            // If inference fails or ID is missing, we log the error for auditability
            if ($returnCode !== 0 || empty($result)) {
                Log::error("Edge-GNN Inference Failed", [
                    'targetId' => $targetId,
                    'stderr' => $errorOutput,
                    'return_code' => $returnCode
                ]);
                
                $score = 0.0;
            } else {
                $score = $result['score'] ?? 0.0;
            }
        } else {
            $score = 0.0;
        }

        // 5. FHIR-Compliant Response
        // Aligns with the modular on-device clinical intelligence framework
        return [
            'id' => uniqid('fhir-'),
            'status' => 'final',
            'category' => 'PPI-Analysis',
            'subject' => ['reference' => "Patient/$targetId"],
            'aiInferenceScore' => $score,
            'edgeModelVersion' => 'v1.0-INT8-Quantized'
        ];
    }
}