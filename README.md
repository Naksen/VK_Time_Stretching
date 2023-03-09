# VK_Time_Stretching
Тестовое задания для стажировки в VK, реализация алгоритма для сжатия / растяжения аудио-файла без изменения тона: https://www.guitarpitchshifter.com/algorithm.html
## Запуск
- запустить bash-скрипт `run.sh`: `run.sh input_file_path output_file_path time_stretch_ratio`
- 0 < `time_stretch_ratio` < 1 -> squeezing
- 1 <= `time_stretch_ratio` -> stretching
- формат аудио-файла - `wav` mono/stereo  
