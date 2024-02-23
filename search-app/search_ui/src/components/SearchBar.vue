<script setup>
import { ref, watch } from 'vue'
// Optionally import default styling
import SearchInput from 'vue-search-input'
import 'vue-search-input/dist/styles.css'

const searchVal = ref('');
const searchResults = ref(null);
async function searchProducts() {
  const searchTerm = searchVal.value
  const response = await fetch("http://localhost:9011/search",{
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*"
    },
    body: JSON.stringify({"search_text": searchTerm, "limit": 1000})
  });
  const data = await response.json();
  searchResults.value = data;
  console.log(data);
  return data
}


</script>
<template>
  <div class="container search-app">
    <SearchInput v-model="searchVal" @keyup.enter="searchProducts"/>
    <div class="divider"></div>
  </div>
  <div v-for="(value, key) in searchResults.results" :key="key">
    <h3><p>{{ value.payload.product_name }}</p></h3>
    
    <p>{{ value.payload.short_product_description }}</p>
  </div>
</template>
<style scoped>
.search-app {
  display: flex;
  width: 100%;
  height: 100%;
}

.search-bar {
  display: flex;
  justify-content: left;
  align-items: left;
  width: 75%;
  height: 1.5rem;
}





/* SearchInput {
  width: 75%;
  padding: 10px;
  font-size: 1.5rem;
  border: 1px solid #ccc;
  border-radius: 5px;
} */
</style>