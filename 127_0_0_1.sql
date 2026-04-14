-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- 主机： 127.0.0.1:3306
-- 生成日期： 2026-04-12 07:51:44
-- 服务器版本： 9.1.0
-- PHP 版本： 8.3.14

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- 数据库： `web_medic`
--
CREATE DATABASE IF NOT EXISTS `web_medic` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
USE `web_medic`;

-- --------------------------------------------------------

--
-- 表的结构 `assessments`
--

DROP TABLE IF EXISTS `assessments`;
CREATE TABLE IF NOT EXISTS `assessments` (
  `userID` varchar(20) NOT NULL,
  `assessment_date` date NOT NULL,
  `height` double NOT NULL,
  `weight` double NOT NULL,
  `bmi` double NOT NULL,
  `health_status` varchar(10) NOT NULL
) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- 转存表中的数据 `assessments`
--

INSERT INTO `assessments` (`userID`, `assessment_date`, `height`, `weight`, `bmi`, `health_status`) VALUES
('5201021999111111X', '2026-03-21', 172, 80, 22, '健康');

-- --------------------------------------------------------

--
-- 表的结构 `user`
--

DROP TABLE IF EXISTS `user`;
CREATE TABLE IF NOT EXISTS `user` (
  `userID` varchar(20) NOT NULL,
  `userName` varchar(20) NOT NULL,
  `userBirth` date NOT NULL,
  `userTel` varchar(20) NOT NULL,
  `created_at` date NOT NULL
) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- 转存表中的数据 `user`
--

INSERT INTO `user` (`userID`, `userName`, `userBirth`, `userTel`, `created_at`) VALUES
('5201021999111111X', 'whm', '1999-03-05', '18185017670', '2026-03-22'),
('52010114440222000', 'lsq', '2026-03-06', '1818000', '2026-03-30');
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
