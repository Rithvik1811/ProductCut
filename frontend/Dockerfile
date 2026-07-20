FROM node:20-alpine

WORKDIR /app

# NEXT_PUBLIC_* vars are baked into the bundle at build time
ARG NEXT_PUBLIC_WS_BASE_URL
ENV NEXT_PUBLIC_WS_BASE_URL=$NEXT_PUBLIC_WS_BASE_URL

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

EXPOSE 3000
ENV PORT=3000
CMD ["npm", "start"]
