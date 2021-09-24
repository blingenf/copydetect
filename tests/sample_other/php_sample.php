<!DOCTYPE HTML>
<html>
<head>
  <meta charset="UTF-8"/>
  <title>Test Document</title>
</head>
<body>
<?php
  session_start();
  if ($_SESSION['logged_in'])
    echo '<a href="logout.php">Sign Out</a>';
  else
    echo '<a href="login.php">Sign In</a>';

  $var1 = 1;
  $var5 = $var1 + 4;
?>
</body>
</html>
