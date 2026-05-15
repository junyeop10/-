Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
pythonwPath = scriptDir & "\.venv\Scripts\pythonw.exe"
appPath = scriptDir & "\app_gui.py"

If fso.FileExists(pythonwPath) Then
  shell.Run """" & pythonwPath & """ """ & appPath & """", 0, False
Else
  shell.Run "pyw """ & appPath & """", 0, False
End If
