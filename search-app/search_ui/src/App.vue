<template>
  <html lang="en">

  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
  </head>

  <body>
    <nav class="navbar bg-body-tertiary">
      <div class="container-fluid">
        <span class="navbar-brand mb-0 h1 app-heading">Neural Search Demo</span>
      </div>
    </nav>
    <div class="app-container">
      <SearchBar />
    </div>
  </body>

  </html>
</template>

<script setup>
import SearchBar from "./components/SearchBar.vue";
</script>


<style scoped>
body {
  font-family: Arial, sans-serif;
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}
.app-heading {
  font-size: 1.5rem;
  color: #4694dd;
  padding: 20px;
  font-weight: bold;   
}
</style>
